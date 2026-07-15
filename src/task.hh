#pragma once

#include <chrono>
#include <coroutine>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <utility>

namespace coropulse {

class Scheduler;
class TickContext;

struct tickDone {};

namespace detail {

inline thread_local TickContext* current_tick_context = nullptr;

inline TickContext& currentTickContext() {
    if (!current_tick_context) {
        throw std::runtime_error("tick context is not active");
    }
    return *current_tick_context;
}

class ScopedTickContext {
public:
    explicit ScopedTickContext(TickContext* ctx) noexcept
        : previous_(current_tick_context) {
        current_tick_context = ctx;
    }

    ~ScopedTickContext() {
        current_tick_context = previous_;
    }

    ScopedTickContext(const ScopedTickContext&) = delete;
    ScopedTickContext& operator=(const ScopedTickContext&) = delete;

private:
    TickContext* previous_;
};

} // namespace detail

template <class T>
class Task;

template <>
class Task<void> {
public:
    struct promise_type {
        Scheduler* scheduler = nullptr;
        std::exception_ptr exception;
        TickContext* tick_context = nullptr;
        std::size_t component_id = 0;
        bool profile_active_time = false;
        bool deferred_resume = false;
        bool tick_done = false;
        std::chrono::nanoseconds active_time{0};

        Task get_return_object() noexcept {
            return Task(handle_type::from_promise(*this));
        }

        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        auto yield_value(tickDone) noexcept {
            struct TickBoundaryAwaiter {
                bool await_ready() const noexcept { return false; }

                void await_suspend(std::coroutine_handle<promise_type> handle) const noexcept {
                    handle.promise().tick_done = true;
                }

                void await_resume() const noexcept {}
            };

            return TickBoundaryAwaiter{};
        }
        void return_void() noexcept {}
        void unhandled_exception() noexcept { exception = std::current_exception(); }
    };

    using handle_type = std::coroutine_handle<promise_type>;

    Task() = default;
    explicit Task(handle_type handle) noexcept : handle_(handle) {}

    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;

    Task(Task&& other) noexcept : handle_(std::exchange(other.handle_, {})) {}

    Task& operator=(Task&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                handle_.destroy();
            }
            handle_ = std::exchange(other.handle_, {});
        }
        return *this;
    }

    ~Task() {
        if (handle_) {
            handle_.destroy();
        }
    }

    explicit operator bool() const noexcept { return static_cast<bool>(handle_); }

private:
    friend class Scheduler;

    void bind(Scheduler& scheduler) {
        if (!handle_) {
            throw std::runtime_error("cannot bind an empty task");
        }
        handle_.promise().scheduler = &scheduler;
    }

    handle_type release() noexcept {
        return std::exchange(handle_, {});
    }

    handle_type handle_{};
};

} // namespace coropulse
