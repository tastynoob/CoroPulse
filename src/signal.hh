#pragma once

#include "runtime.hh"

#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace coropulse {

template <class T>
class Signal final : public TickObject {
public:
    class Master;
    class Slaver;
    class Awaiter;

    explicit Signal(std::string name = {}) : name_(std::move(name)) {}

    Master master() {
        if (master_created_) {
            throw std::runtime_error(objectError("signal", name_, "master already exists"));
        }
        master_created_ = true;
        return Master(this);
    }

    Slaver addSlaver() {
        const auto id = slaver_count_++;
        read_by_.push_back(false);
        return Slaver(this, id);
    }

    void beginTick(TickId tick) override {
        std::lock_guard lock(mutex_);
        tick_ = tick;
        ready_ = false;
        value_.reset();
        waiters_.clear();
        read_by_.assign(slaver_count_, false);
    }

    void commit(TickId) override {}

    void endTick(TickId) override {
        std::lock_guard lock(mutex_);
        if (!waiters_.empty()) {
            throw std::runtime_error(
                objectError("signal", name_, "tick ended with pending waiters"));
        }

        if (!ready_) {
            return;
        }

        for (bool read : read_by_) {
            if (!read) {
                throw std::runtime_error(
                    objectError("signal", name_, "set value was not read by every slaver"));
            }
        }
    }

private:
    friend class Master;
    friend class Slaver;
    friend class Awaiter;

    bool allReadLocked() const {
        for (bool read : read_by_) {
            if (!read) {
                return false;
            }
        }
        return true;
    }

    void set(TickContext& ctx, T value) {
        std::vector<Scheduler::Handle> waiters;

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
            value_ = std::move(value);
            waiters = std::move(waiters_);
            waiters_.clear();
        }

        for (auto waiter : waiters) {
            ctx.scheduler().schedule(waiter);
        }
    }

    bool awaitReady(SlaverId id, TickContext& ctx) {
        std::lock_guard lock(mutex_);
        checkSlaverLocked(id);
        if (ctx.tick() != tick_) {
            throw std::runtime_error(objectError("signal", name_, "read with stale context"));
        }
        if (read_by_[id]) {
            throw std::runtime_error(
                objectError("signal", name_, "duplicate slaver read in one tick"));
        }
        return ready_;
    }

    bool awaitSuspend(SlaverId id, TickContext& ctx, Scheduler::Handle handle) {
        std::lock_guard lock(mutex_);
        checkSlaverLocked(id);
        if (ctx.tick() != tick_) {
            throw std::runtime_error(objectError("signal", name_, "read with stale context"));
        }
        if (read_by_[id]) {
            throw std::runtime_error(
                objectError("signal", name_, "duplicate slaver read in one tick"));
        }
        if (ready_) {
            return false;
        }
        waiters_.push_back(handle);
        return true;
    }

    T awaitResume(SlaverId id) {
        std::lock_guard lock(mutex_);
        checkSlaverLocked(id);
        if (!ready_ || !value_) {
            throw std::runtime_error(objectError("signal", name_, "resumed before set"));
        }
        if (read_by_[id]) {
            throw std::runtime_error(
                objectError("signal", name_, "duplicate slaver read in one tick"));
        }

        auto value = *value_;
        read_by_[id] = true;

        if (allReadLocked()) {
            value_.reset();
        }

        return value;
    }

    void checkSlaverLocked(SlaverId id) const {
        if (id >= slaver_count_) {
            throw std::runtime_error(objectError("signal", name_, "invalid slaver id"));
        }
    }

    std::string name_;
    TickId tick_ = 0;
    bool master_created_ = false;
    std::size_t slaver_count_ = 0;

    std::mutex mutex_;
    bool ready_ = false;
    std::optional<T> value_;
    std::vector<bool> read_by_;
    std::vector<Scheduler::Handle> waiters_;
};

template <class T>
class Signal<T>::Master {
public:
    explicit Master(Signal* signal = nullptr) noexcept : signal_(signal) {}

    void set(TickContext& ctx, T value) const {
        ensure();
        signal_->set(ctx, std::move(value));
    }

private:
    void ensure() const {
        if (!signal_) {
            throw std::runtime_error("empty signal master");
        }
    }

    Signal* signal_;
};

template <class T>
class Signal<T>::Slaver {
public:
    Slaver(Signal* signal = nullptr, SlaverId id = 0) noexcept
        : signal_(signal), id_(id) {}

    Awaiter read(TickContext& ctx) const {
        ensure();
        return Awaiter(signal_, id_, &ctx);
    }

private:
    void ensure() const {
        if (!signal_) {
            throw std::runtime_error("empty signal slaver");
        }
    }

    Signal* signal_;
    SlaverId id_;
};

template <class T>
class Signal<T>::Awaiter {
public:
    Awaiter(Signal* signal, SlaverId id, TickContext* ctx) noexcept
        : signal_(signal), id_(id), ctx_(ctx) {}

    bool await_ready() {
        return signal_->awaitReady(id_, *ctx_);
    }

    bool await_suspend(Scheduler::Handle handle) {
        return signal_->awaitSuspend(id_, *ctx_, handle);
    }

    T await_resume() {
        return signal_->awaitResume(id_);
    }

private:
    Signal* signal_;
    SlaverId id_;
    TickContext* ctx_;
};

} // namespace coropulse
