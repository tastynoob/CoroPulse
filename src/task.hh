#pragma once

#include <chrono>
#include <coroutine>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <utility>

namespace coropulse {

class Scheduler;

template <class T>
class Task;

template <>
class Task<void> {
public:
    struct promise_type {
        Scheduler* scheduler = nullptr;
        std::exception_ptr exception;
        std::size_t component_id = 0;
        bool profile_active_time = false;
        std::chrono::nanoseconds active_time{0};

        Task get_return_object() noexcept {
            return Task(handle_type::from_promise(*this));
        }

        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
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
