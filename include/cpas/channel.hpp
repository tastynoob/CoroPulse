#pragma once

#include "cpas/runtime.hpp"

#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cpas {

template <class T>
class Channel final : public TickObject {
public:
    class Master;
    class Slaver;

    explicit Channel(std::string name = {}) : name_(std::move(name)) {}

    Master master() {
        if (master_created_) {
            throw std::runtime_error(objectError("channel", name_, "master already exists"));
        }
        master_created_ = true;
        return Master(this);
    }

    Slaver addSlaver() {
        const auto id = slaver_count_++;
        if (visible_) {
            visible_->consumed_by.push_back(false);
        }
        return Slaver(this, id);
    }

    void beginTick(TickId tick) override {
        std::lock_guard lock(mutex_);
        tick_ = tick;
        pending_write_.reset();

        discardConsumedVisibleLocked();

        if (!visible_ && next_) {
            visible_ = Entry{std::move(*next_), tick_, std::vector<bool>(slaver_count_, false)};
            next_.reset();
        }

        discardConsumedVisibleLocked();
        write_open_at_tick_start_ = !visible_ && !next_;
    }

    void commit(TickId tick) override {
        std::lock_guard lock(mutex_);
        if (tick != tick_) {
            throw std::runtime_error(objectError("channel", name_, "commit with stale tick"));
        }

        discardConsumedVisibleLocked();

        if (!pending_write_) {
            return;
        }

        if (visible_ || next_) {
            throw std::runtime_error(
                objectError("channel", name_, "commit would overflow the single slot"));
        }

        next_ = std::move(pending_write_);
        pending_write_.reset();
    }

    void endTick(TickId) override {
        std::lock_guard lock(mutex_);
        discardConsumedVisibleLocked();
    }

private:
    struct Entry {
        T value;
        TickId visible_tick;
        std::vector<bool> consumed_by;
    };

    friend class Master;
    friend class Slaver;

    bool canWrite(TickContext& ctx) const {
        std::lock_guard lock(mutex_);
        if (ctx.tick() != tick_) {
            throw std::runtime_error(objectError("channel", name_, "write with stale context"));
        }
        return write_open_at_tick_start_ && !pending_write_;
    }

    bool write(TickContext& ctx, T value) {
        std::lock_guard lock(mutex_);
        if (ctx.tick() != tick_) {
            throw std::runtime_error(objectError("channel", name_, "write with stale context"));
        }
        if (!write_open_at_tick_start_ || pending_write_) {
            return false;
        }
        pending_write_ = std::move(value);
        return true;
    }

    std::optional<T> read(TickContext& ctx, SlaverId id) {
        std::lock_guard lock(mutex_);
        if (ctx.tick() != tick_) {
            throw std::runtime_error(objectError("channel", name_, "read with stale context"));
        }
        if (id >= slaver_count_) {
            throw std::runtime_error(objectError("channel", name_, "invalid slaver id"));
        }
        if (!visible_) {
            return std::nullopt;
        }
        if (visible_->consumed_by[id]) {
            return std::nullopt;
        }

        visible_->consumed_by[id] = true;
        return visible_->value;
    }

    bool hasValue(TickContext& ctx, SlaverId id) const {
        std::lock_guard lock(mutex_);
        if (ctx.tick() != tick_) {
            throw std::runtime_error(objectError("channel", name_, "peek with stale context"));
        }
        if (id >= slaver_count_) {
            throw std::runtime_error(objectError("channel", name_, "invalid slaver id"));
        }
        return visible_ && !visible_->consumed_by[id];
    }

    bool visibleConsumedLocked() const {
        if (!visible_) {
            return true;
        }
        for (bool consumed : visible_->consumed_by) {
            if (!consumed) {
                return false;
            }
        }
        return true;
    }

    void discardConsumedVisibleLocked() {
        if (visibleConsumedLocked()) {
            visible_.reset();
        }
    }

    std::string name_;
    TickId tick_ = 0;
    bool master_created_ = false;
    std::size_t slaver_count_ = 0;

    mutable std::mutex mutex_;
    bool write_open_at_tick_start_ = true;
    std::optional<Entry> visible_;
    std::optional<T> next_;
    std::optional<T> pending_write_;
};

template <class T>
class Channel<T>::Master {
public:
    explicit Master(Channel* channel = nullptr) noexcept : channel_(channel) {}

    bool canWrite(TickContext& ctx) const {
        ensure();
        return channel_->canWrite(ctx);
    }

    bool write(TickContext& ctx, const T& value) const {
        ensure();
        return channel_->write(ctx, value);
    }

    bool write(TickContext& ctx, T&& value) const {
        ensure();
        return channel_->write(ctx, std::move(value));
    }

private:
    void ensure() const {
        if (!channel_) {
            throw std::runtime_error("empty channel master");
        }
    }

    Channel* channel_;
};

template <class T>
class Channel<T>::Slaver {
public:
    Slaver(Channel* channel = nullptr, SlaverId id = 0) noexcept
        : channel_(channel), id_(id) {}

    bool hasValue(TickContext& ctx) const {
        ensure();
        return channel_->hasValue(ctx, id_);
    }

    std::optional<T> read(TickContext& ctx) const {
        ensure();
        return channel_->read(ctx, id_);
    }

private:
    void ensure() const {
        if (!channel_) {
            throw std::runtime_error("empty channel slaver");
        }
    }

    Channel* channel_;
    SlaverId id_;
};

} // namespace cpas
