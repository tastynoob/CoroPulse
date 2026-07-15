#pragma once

#include "task.hh"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace coropulse {

class Scheduler {
public:
    using Handle = Task<void>::handle_type;
    using Duration = std::chrono::nanoseconds;

    struct TaskSample {
        std::size_t component_id = 0;
        Duration active_time{0};
    };

    explicit Scheduler(std::size_t worker_count = 1)
        : worker_count_(worker_count == 0 ? 1 : worker_count) {
        workers_.reserve(worker_count_);
        for (std::size_t i = 0; i < worker_count_; ++i) {
            workers_.emplace_back([this] { workerLoop(); });
        }
    }

    ~Scheduler() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
            dispatching_ = false;
        }
        work_cv_.notify_all();

        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }

        for (auto handle : owned_) {
            if (handle) {
                handle.destroy();
            }
        }
    }

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    void addProcess(Task<void> task, std::size_t component_id = 0,
                    TickContext* tick_context = nullptr) {
        task.bind(*this);
        auto handle = task.release();
        handle.promise().tick_context = tick_context;
        handle.promise().component_id = component_id;
        handle.promise().profile_active_time = false;
        handle.promise().deferred_resume = false;
        handle.promise().persistent_wait = false;
        handle.promise().persistent_wait_counted = false;
        handle.promise().tick_done = false;
        handle.promise().dispatch_epoch = dispatch_epoch_;
        handle.promise().active_time = Duration{0};

        std::lock_guard lock(mutex_);
        if (failed_) {
            handle.destroy();
            throw std::runtime_error("cannot add task to a failed scheduler");
        }
        if (dispatching_ || active_ != 0) {
            handle.destroy();
            throw std::runtime_error("cannot add task while scheduler is running");
        }
        if (live_ == 0 && ready_.empty() && next_tick_count_ == 0) {
            owned_.clear();
        }
        owned_.push_back(handle);
        if (component_id >= next_tick_.size()) {
            next_tick_.resize(component_id + 1);
        }
        next_tick_[component_id] = handle;
        ++next_tick_count_;
        ++live_;
    }

    void schedule(Handle handle) {
        if (!handle) {
            throw std::runtime_error("cannot schedule an empty coroutine handle");
        }

        {
            std::lock_guard lock(mutex_);
            if (stopping_ || !dispatching_ || first_exception_) {
                return;
            }
            if (handle.promise().persistent_wait) {
                handle.promise().persistent_wait = false;
                if (handle.promise().persistent_wait_counted) {
                    handle.promise().persistent_wait_counted = false;
                    --sleeping_count_;
                }
            }
            if (handle.promise().dispatch_epoch != dispatch_epoch_) {
                prepareForCurrentTickLocked(handle);
            }
            ready_.push_back(handle);
        }
        work_cv_.notify_one();
    }

    void defer(Handle handle) {
        if (!handle) {
            throw std::runtime_error("cannot defer an empty coroutine handle");
        }

        handle.promise().deferred_resume = true;
    }

    void sleep(Handle handle) {
        if (!handle) {
            throw std::runtime_error("cannot sleep an empty coroutine handle");
        }

        std::lock_guard lock(mutex_);
        handle.promise().persistent_wait = true;
        handle.promise().persistent_wait_counted = false;
    }

    void run(const std::vector<std::size_t>& component_order,
             bool profile_active_time) {
        std::unique_lock lock(mutex_);
        if (failed_) {
            throw std::runtime_error("cannot reuse scheduler after failed tick");
        }
        if (dispatching_ || active_ != 0) {
            throw std::runtime_error("scheduler is already running");
        }

        startTickLocked(component_order, profile_active_time);
        dispatching_ = true;
        work_cv_.notify_all();

        while (true) {
            if (first_exception_) {
                failed_ = true;
                dispatching_ = false;
                work_cv_.notify_all();
                done_cv_.wait(lock, [this] { return active_ == 0; });

                auto exception = first_exception_;
                lock.unlock();
                std::rethrow_exception(exception);
            }

            if (active_ == 0 && ready_.empty() &&
                live_ == next_tick_count_ + sleeping_count_) {
                dispatching_ = false;
                work_cv_.notify_all();
                return;
            }

            if (ready_.empty() && active_ == 0) {
                failed_ = true;
                dispatching_ = false;
                first_exception_ = std::make_exception_ptr(std::runtime_error(
                    "deadlock: current tick drained while coroutine(s) are still waiting"));
                auto exception = first_exception_;
                lock.unlock();
                std::rethrow_exception(exception);
            }

            done_cv_.wait(lock, [this] {
                return first_exception_ || (ready_.empty() && active_ == 0);
            });
        }
    }

    std::size_t live() const {
        std::lock_guard lock(mutex_);
        return live_;
    }

    std::size_t workerCount() const noexcept { return worker_count_; }

    std::vector<TaskSample> takeSamples() {
        std::lock_guard lock(mutex_);
        std::vector<TaskSample> samples;
        samples.swap(samples_);
        return samples;
    }

private:
    using Clock = std::chrono::steady_clock;

    void startTickLocked(const std::vector<std::size_t>& component_order,
                         bool profile_active_time) {
        if (!ready_.empty()) {
            throw std::runtime_error("scheduler ready queue is not empty at tick start");
        }

        samples_.clear();
        current_profile_active_time_ = profile_active_time;
        ++dispatch_epoch_;
        for (const auto component_id : component_order) {
            if (component_id >= next_tick_.size()) {
                continue;
            }

            auto handle = next_tick_[component_id];
            if (!handle) {
                continue;
            }

            next_tick_[component_id] = {};
            --next_tick_count_;
            ready_.push_back(handle);
        }

        for (auto handle : ready_) {
            prepareForCurrentTickLocked(handle);
        }
    }

    void prepareForCurrentTickLocked(Handle handle) {
        handle.promise().profile_active_time = current_profile_active_time_;
        handle.promise().deferred_resume = false;
        handle.promise().persistent_wait = false;
        handle.promise().persistent_wait_counted = false;
        handle.promise().tick_done = false;
        handle.promise().dispatch_epoch = dispatch_epoch_;
        handle.promise().active_time = Duration{0};
    }

    void workerLoop() noexcept {
        for (;;) {
            Handle handle{};

            {
                std::unique_lock lock(mutex_);
                work_cv_.wait(lock, [this] {
                    return stopping_ || (dispatching_ && !first_exception_ && !ready_.empty());
                });

                if (stopping_) {
                    return;
                }

                handle = ready_.front();
                ready_.pop_front();
                ++active_;
            }

            runOne(handle);
        }
    }

    void runOne(Handle handle) noexcept {
        std::exception_ptr resume_exception;

        if (handle.promise().profile_active_time) {
            const auto start = Clock::now();
            try {
                detail::ScopedTickContext context(handle.promise().tick_context);
                handle.resume();
            } catch (...) {
                resume_exception = std::current_exception();
            }
            const auto elapsed =
                std::chrono::duration_cast<Duration>(Clock::now() - start);
            handle.promise().active_time += elapsed;
        } else {
            try {
                detail::ScopedTickContext context(handle.promise().tick_context);
                handle.resume();
            } catch (...) {
                resume_exception = std::current_exception();
            }
        }

        std::exception_ptr coroutine_exception;
        TaskSample sample;
        bool has_sample = false;
        const bool done = !resume_exception && handle.done();
        const bool should_defer = !resume_exception && !done &&
                                  handle.promise().deferred_resume;
        const bool tick_done = !resume_exception && !done &&
                               handle.promise().tick_done;
        handle.promise().deferred_resume = false;
        handle.promise().tick_done = false;
        if (done) {
            coroutine_exception = handle.promise().exception;
            if (!coroutine_exception) {
                coroutine_exception = std::make_exception_ptr(std::runtime_error(
                    "component process returned unexpectedly: component_id=" +
                    std::to_string(handle.promise().component_id)));
            }
            if (handle.promise().profile_active_time) {
                sample = TaskSample{
                    handle.promise().component_id,
                    handle.promise().active_time,
                };
                has_sample = true;
            }
        }

        {
            std::lock_guard lock(mutex_);
            --active_;

            if (done) {
                if (has_sample) {
                    samples_.push_back(sample);
                }
                handle.destroy();
                forgetLocked(handle);
                --live_;
            } else if (tick_done && !stopping_ && dispatching_ && !first_exception_) {
                if (handle.promise().profile_active_time) {
                    samples_.push_back(TaskSample{
                        handle.promise().component_id,
                        handle.promise().active_time,
                    });
                }
                const auto component_id = handle.promise().component_id;
                if (component_id >= next_tick_.size()) {
                    next_tick_.resize(component_id + 1);
                }
                if (!next_tick_[component_id]) {
                    ++next_tick_count_;
                }
                next_tick_[component_id] = handle;
            } else if (should_defer && !stopping_ && dispatching_ && !first_exception_) {
                ready_.push_back(handle);
            } else if (handle.promise().persistent_wait && !stopping_ && dispatching_ &&
                       !first_exception_) {
                if (!handle.promise().persistent_wait_counted) {
                    handle.promise().persistent_wait_counted = true;
                    ++sleeping_count_;
                }
            }

            if (resume_exception && !first_exception_) {
                first_exception_ = resume_exception;
            } else if (coroutine_exception && !first_exception_) {
                first_exception_ = coroutine_exception;
            }
        }

        done_cv_.notify_all();
        work_cv_.notify_all();
    }

    void forgetLocked(Handle handle) {
        for (auto& owned : owned_) {
            if (owned == handle) {
                owned = {};
                return;
            }
        }
    }

    const std::size_t worker_count_;
    std::vector<std::thread> workers_;
    mutable std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;
    std::deque<Handle> ready_;
    std::vector<Handle> next_tick_;
    std::vector<Handle> owned_;
    std::vector<TaskSample> samples_;
    std::size_t live_ = 0;
    std::size_t active_ = 0;
    std::size_t next_tick_count_ = 0;
    std::size_t sleeping_count_ = 0;
    std::size_t dispatch_epoch_ = 0;
    bool current_profile_active_time_ = false;
    bool dispatching_ = false;
    bool stopping_ = false;
    bool failed_ = false;
    std::exception_ptr first_exception_;
};

} // namespace coropulse
