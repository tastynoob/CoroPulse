#pragma once

#include "cpas/task.hpp"

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace cpas {

class Scheduler {
public:
    using Handle = Task<void>::handle_type;

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

    void add(Task<void> task) {
        task.bind(*this);
        auto handle = task.release();

        std::lock_guard lock(mutex_);
        if (failed_) {
            handle.destroy();
            throw std::runtime_error("cannot add task to a failed scheduler");
        }
        if (dispatching_ || active_ != 0) {
            handle.destroy();
            throw std::runtime_error("cannot add task while scheduler is running");
        }
        if (live_ == 0 && ready_.empty()) {
            owned_.clear();
        }
        owned_.push_back(handle);
        ready_.push_back(handle);
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
            ready_.push_back(handle);
        }
        work_cv_.notify_one();
    }

    void run() {
        std::unique_lock lock(mutex_);
        if (failed_) {
            throw std::runtime_error("cannot reuse scheduler after failed tick");
        }
        if (dispatching_ || active_ != 0) {
            throw std::runtime_error("scheduler is already running");
        }

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

            if (live_ == 0) {
                dispatching_ = false;
                work_cv_.notify_all();
                return;
            }

            if (ready_.empty() && active_ == 0) {
                failed_ = true;
                dispatching_ = false;
                first_exception_ = std::make_exception_ptr(std::runtime_error(
                    "deadlock: PEQ drained while coroutine(s) are still waiting"));
                auto exception = first_exception_;
                lock.unlock();
                std::rethrow_exception(exception);
            }

            done_cv_.wait(lock);
        }
    }

    std::size_t live() const {
        std::lock_guard lock(mutex_);
        return live_;
    }

    std::size_t workerCount() const noexcept { return worker_count_; }

private:
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

        try {
            handle.resume();
        } catch (...) {
            resume_exception = std::current_exception();
        }

        std::exception_ptr coroutine_exception;
        const bool done = !resume_exception && handle.done();
        if (done) {
            coroutine_exception = handle.promise().exception;
        }

        {
            std::lock_guard lock(mutex_);
            --active_;

            if (done) {
                handle.destroy();
                forgetLocked(handle);
                --live_;
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
    std::vector<Handle> owned_;
    std::size_t live_ = 0;
    std::size_t active_ = 0;
    bool dispatching_ = false;
    bool stopping_ = false;
    bool failed_ = false;
    std::exception_ptr first_exception_;
};

} // namespace cpas
