#pragma once

#include "runtime.hh"

#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace coropulse {

class Simulator;
template <class T = void>
class Signal;
template <class T = void>
class SignalOutput;
template <class T = void>
class SignalInput;

namespace detail {

template <class T>
class SignalPayload {
public:
    static constexpr bool require_all_read = true;

    void reset() {
        value_.reset();
    }

    void set(T value) {
        value_ = std::move(value);
    }

    bool ready() const noexcept {
        return value_.has_value();
    }

    T read(const std::string& name) const {
        if (!value_) {
            throw std::runtime_error(objectError("signal", name, "resumed before set"));
        }
        return *value_;
    }

    void clear() {
        value_.reset();
    }

private:
    std::optional<T> value_;
};

template <>
class SignalPayload<void> {
public:
    static constexpr bool require_all_read = false;

    void reset() noexcept {}
    void set() noexcept {}
    bool ready() const noexcept { return true; }
    void read(const std::string&) const noexcept {}
    void clear() noexcept {}
};

} // namespace detail

template <class T>
class Signal final : public TickObject {
public:
    explicit Signal(std::string name = {}) : name_(std::move(name)) {}

private:
    class Awaiter;

    friend class Simulator;
    friend class SignalOutput<T>;
    friend class SignalInput<T>;
    friend class Awaiter;

    void attachWriter() {
        if (writer_connected_) {
            throw std::runtime_error(objectError("signal", name_, "writer already exists"));
        }
        writer_connected_ = true;
    }

    std::size_t attachReader() {
        const auto id = reader_count_++;
        read_by_.push_back(false);
        return id;
    }

    void beginTick(TickId tick) override {
        tick_ = tick;
        ready_ = false;
        payload_.reset();
        read_waiters_.clear();
        read_by_.assign(reader_count_, false);
    }

    void commit(TickId) override {}

    void endTick(TickId) override {
        if (!read_waiters_.empty()) {
            throw std::runtime_error(
                objectError("signal", name_, "tick ended with pending waiters"));
        }

        if constexpr (!Payload::require_all_read) {
            return;
        }

        if (!ready_) {
            return;
        }

        for (bool read : read_by_) {
            if (!read) {
                throw std::runtime_error(
                    objectError("signal", name_, "set value was not read by every reader"));
            }
        }
    }

    void set(TickContext& ctx) requires std::is_void_v<T> {
        setReady(ctx);
    }

    template <class U = T>
    void set(TickContext& ctx, U value) requires(!std::is_void_v<T>) {
        setReady(ctx, std::move(value));
    }

    template <class... Args>
    void setReady(TickContext& ctx, Args&&... args) {
        std::vector<Scheduler::Handle> waiters;
        Scheduler::Handle single_waiter{};

        {
            std::lock_guard lock(mutex_);
            if (ctx.tick() != tick_) {
                throw std::runtime_error(objectError("signal", name_, "set with stale context"));
            }
            if (ready_) {
                throw std::runtime_error(
                    objectError("signal", name_, "duplicate set in one tick"));
            }

            ready_ = true;
            payload_.set(std::forward<Args>(args)...);
            const auto waiter_count = read_waiters_.size() + wait_waiters_.size();
            if (waiter_count == 1) {
                if (!read_waiters_.empty()) {
                    single_waiter = read_waiters_.front();
                    read_waiters_.clear();
                } else {
                    single_waiter = wait_waiters_.front();
                    wait_waiters_.clear();
                }
            } else if (waiter_count > 1) {
                waiters.reserve(waiter_count);
                waiters.insert(waiters.end(), read_waiters_.begin(), read_waiters_.end());
                waiters.insert(waiters.end(), wait_waiters_.begin(), wait_waiters_.end());
                read_waiters_.clear();
                wait_waiters_.clear();
            }
        }

        if (single_waiter) {
            ctx.scheduler().schedule(single_waiter);
        } else {
            ctx.scheduler().scheduleMany(std::move(waiters));
        }
    }

    bool awaitReady(std::size_t reader_id, TickContext& ctx) {
        std::lock_guard lock(mutex_);
        checkReaderLocked(reader_id);
        if (ctx.tick() != tick_) {
            throw std::runtime_error(objectError("signal", name_, "read with stale context"));
        }
        if (read_by_[reader_id]) {
            throw std::runtime_error(
                objectError("signal", name_, "duplicate reader read in one tick"));
        }
        return ready_;
    }

    bool awaitSuspend(std::size_t reader_id, TickContext& ctx, Scheduler::Handle handle,
                      bool persistent) {
        std::lock_guard lock(mutex_);
        checkReaderLocked(reader_id);
        if (ctx.tick() != tick_) {
            throw std::runtime_error(objectError("signal", name_, "read with stale context"));
        }
        if (read_by_[reader_id]) {
            throw std::runtime_error(
                objectError("signal", name_, "duplicate reader read in one tick"));
        }
        if (ready_) {
            return false;
        }
        if (persistent) {
            ctx.scheduler().sleep(handle);
            wait_waiters_.push_back(handle);
        } else {
            read_waiters_.push_back(handle);
        }
        return true;
    }

    auto awaitResume(std::size_t reader_id) {
        std::lock_guard lock(mutex_);
        checkReaderLocked(reader_id);
        if (!ready_) {
            throw std::runtime_error(objectError("signal", name_, "resumed before set"));
        }
        if constexpr (!std::is_void_v<T>) {
            if (!payload_.ready()) {
                throw std::runtime_error(objectError("signal", name_, "resumed before set"));
            }
        }
        if (read_by_[reader_id]) {
            throw std::runtime_error(
                objectError("signal", name_, "duplicate reader read in one tick"));
        }

        if constexpr (std::is_void_v<T>) {
            read_by_[reader_id] = true;
            return;
        } else {
            auto value = payload_.read(name_);
            read_by_[reader_id] = true;

            if (allReadLocked()) {
                payload_.clear();
            }

            return value;
        }
    }

    bool allReadLocked() const {
        for (bool read : read_by_) {
            if (!read) {
                return false;
            }
        }
        return true;
    }

    void checkReaderLocked(std::size_t reader_id) const {
        if (reader_id >= reader_count_) {
            throw std::runtime_error(objectError("signal", name_, "invalid reader id"));
        }
    }

    using Payload = detail::SignalPayload<T>;

    std::string name_;
    TickId tick_ = 0;
    bool writer_connected_ = false;
    std::size_t reader_count_ = 0;

    std::mutex mutex_;
    bool ready_ = false;
    Payload payload_;
    std::vector<bool> read_by_;
    std::vector<Scheduler::Handle> read_waiters_;
    std::vector<Scheduler::Handle> wait_waiters_;
};

template <class T>
class Signal<T>::Awaiter {
public:
    Awaiter(Signal* signal, std::size_t reader_id, TickContext* ctx,
            bool persistent) noexcept
        : signal_(signal),
          reader_id_(reader_id),
          ctx_(ctx),
          persistent_(persistent) {}

    bool await_ready() {
        return signal_->awaitReady(reader_id_, *ctx_);
    }

    bool await_suspend(Scheduler::Handle handle) {
        return signal_->awaitSuspend(reader_id_, *ctx_, handle, persistent_);
    }

    auto await_resume() {
        return signal_->awaitResume(reader_id_);
    }

private:
    Signal* signal_;
    std::size_t reader_id_;
    TickContext* ctx_;
    bool persistent_;
};

template <class T>
class SignalOutput {
public:
    explicit SignalOutput(std::string name = {}) : name_(std::move(name)) {}

    bool connected() const noexcept {
        return connected_;
    }

    const std::string& name() const noexcept {
        return name_;
    }

    void set() const requires std::is_void_v<T> {
        ensure();
        signal_->set(detail::currentTickContext());
    }

    template <class U = T>
    void set(U value) const requires(!std::is_void_v<T>) {
        ensure();
        signal_->set(detail::currentTickContext(), std::move(value));
    }

private:
    friend class Simulator;

    void bind(Signal<T>& signal) {
        if (connected_) {
            throw std::runtime_error("signal output port is already connected");
        }
        signal.attachWriter();
        signal_ = &signal;
        connected_ = true;
    }

    void ensure() const {
        if (!connected_) {
            throw std::runtime_error("signal output port is not connected");
        }
    }

    std::string name_;
    Signal<T>* signal_ = nullptr;
    bool connected_ = false;
};

template <class T>
class SignalInput {
public:
    explicit SignalInput(std::string name = {}) : name_(std::move(name)) {}

    bool connected() const noexcept {
        return connected_;
    }

    const std::string& name() const noexcept {
        return name_;
    }

    typename Signal<T>::Awaiter read() const {
        ensure();
        return typename Signal<T>::Awaiter(signal_, reader_id_, &detail::currentTickContext(),
                                           false);
    }

    typename Signal<T>::Awaiter wait() const {
        ensure();
        return typename Signal<T>::Awaiter(signal_, reader_id_, &detail::currentTickContext(),
                                           true);
    }

private:
    friend class Simulator;

    void bind(Signal<T>& signal) {
        if (connected_) {
            throw std::runtime_error("signal input port is already connected");
        }
        signal_ = &signal;
        reader_id_ = signal.attachReader();
        connected_ = true;
    }

    void ensure() const {
        if (!connected_) {
            throw std::runtime_error("signal input port is not connected");
        }
    }

    std::string name_;
    Signal<T>* signal_ = nullptr;
    std::size_t reader_id_ = 0;
    bool connected_ = false;
};

} // namespace coropulse
