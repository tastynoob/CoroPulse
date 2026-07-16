#pragma once

#include "runtime.hh"

#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace coropulse {

class Simulator;
template <class T>
class Output;
template <class T>
class Input;

template <class T>
class ReadRef {
public:
    ReadRef() = default;
    explicit ReadRef(const T* value) noexcept : value_(value) {}

    explicit operator bool() const noexcept { return value_ != nullptr; }
    bool has_value() const noexcept { return value_ != nullptr; }

    const T& operator*() const {
        ensure();
        return *value_;
    }

    const T* operator->() const {
        ensure();
        return value_;
    }

    const T& value() const {
        ensure();
        return *value_;
    }

private:
    void ensure() const {
        if (!value_) {
            throw std::runtime_error("read reference has no value");
        }
    }

    const T* value_ = nullptr;
};

template <class T>
class Channel final : public TickObject {
public:
    explicit Channel(std::string name = {}) : name_(std::move(name)) {}

private:
    friend class Simulator;
    friend class Output<T>;
    friend class Input<T>;

    void attachWriter() {
        if (writer_connected_) {
            throw std::runtime_error(objectError("channel", name_, "writer already exists"));
        }
        writer_connected_ = true;
    }

    std::size_t attachReader() {
        const auto id = reader_count_++;
        if (visible_) {
            visible_->consumed_by.push_back(false);
        }
        return id;
    }

    void beginTick(TickId tick) override {
        tick_ = tick;
        pending_write_.reset();

        discardConsumedVisibleLocked();

        if (!visible_ && next_) {
            visible_ = Entry{std::move(*next_), tick_, std::vector<bool>(reader_count_, false)};
            next_.reset();
        }

        discardConsumedVisibleLocked();
        write_open_at_tick_start_ = !visible_ && !next_;
    }

    void commit(TickId tick) override {
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
        discardConsumedVisibleLocked();
    }

    struct Entry {
        T value;
        TickId visible_tick;
        std::vector<bool> consumed_by;
    };

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

    ReadRef<T> read(TickContext& ctx, std::size_t reader_id) {
        std::lock_guard lock(mutex_);
        if (ctx.tick() != tick_) {
            throw std::runtime_error(objectError("channel", name_, "read with stale context"));
        }
        if (reader_id >= reader_count_) {
            throw std::runtime_error(objectError("channel", name_, "invalid reader id"));
        }
        if (!visible_) {
            return ReadRef<T>{};
        }
        if (visible_->consumed_by[reader_id]) {
            return ReadRef<T>{};
        }

        visible_->consumed_by[reader_id] = true;
        return ReadRef<T>(&visible_->value);
    }

    std::optional<T> take(TickContext& ctx, std::size_t reader_id) {
        std::lock_guard lock(mutex_);
        if (ctx.tick() != tick_) {
            throw std::runtime_error(objectError("channel", name_, "take with stale context"));
        }
        if (reader_id >= reader_count_) {
            throw std::runtime_error(objectError("channel", name_, "invalid reader id"));
        }
        if (reader_count_ != 1) {
            throw std::runtime_error(objectError("channel", name_,
                                                 "take requires exactly one reader"));
        }
        if (!visible_) {
            return std::nullopt;
        }
        if (visible_->consumed_by[reader_id]) {
            return std::nullopt;
        }

        visible_->consumed_by[reader_id] = true;
        auto value = std::move(visible_->value);
        visible_.reset();
        return value;
    }

    bool hasValue(TickContext& ctx, std::size_t reader_id) const {
        std::lock_guard lock(mutex_);
        if (ctx.tick() != tick_) {
            throw std::runtime_error(objectError("channel", name_, "peek with stale context"));
        }
        if (reader_id >= reader_count_) {
            throw std::runtime_error(objectError("channel", name_, "invalid reader id"));
        }
        return visible_ && !visible_->consumed_by[reader_id];
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
    bool writer_connected_ = false;
    std::size_t reader_count_ = 0;

    mutable std::mutex mutex_;
    bool write_open_at_tick_start_ = true;
    std::optional<Entry> visible_;
    std::optional<T> next_;
    std::optional<T> pending_write_;
};

template <class T>
class Output {
public:
    explicit Output(std::string name = {}) : name_(std::move(name)) {}

    bool connected() const noexcept {
        return connected_;
    }

    const std::string& name() const noexcept {
        return name_;
    }

    bool canWrite() const {
        ensure();
        return channel_->canWrite(detail::currentTickContext());
    }

    bool write(const T& value) const {
        ensure();
        return channel_->write(detail::currentTickContext(), value);
    }

    bool write(T&& value) const {
        ensure();
        return channel_->write(detail::currentTickContext(), std::move(value));
    }

private:
    friend class Simulator;

    void bind(Channel<T>& channel) {
        if (connected_) {
            throw std::runtime_error("output port is already connected");
        }
        channel.attachWriter();
        channel_ = &channel;
        connected_ = true;
    }

    void ensure() const {
        if (!connected_) {
            throw std::runtime_error("output port is not connected");
        }
    }

    std::string name_;
    Channel<T>* channel_ = nullptr;
    bool connected_ = false;
};

template <class T>
class Input {
public:
    explicit Input(std::string name = {}) : name_(std::move(name)) {}

    bool connected() const noexcept {
        return connected_;
    }

    const std::string& name() const noexcept {
        return name_;
    }

    bool hasValue() const {
        ensure();
        return channel_->hasValue(detail::currentTickContext(), reader_id_);
    }

    ReadRef<T> read() const {
        ensure();
        return channel_->read(detail::currentTickContext(), reader_id_);
    }

    std::optional<T> take() const {
        ensure();
        return channel_->take(detail::currentTickContext(), reader_id_);
    }

private:
    friend class Simulator;

    void bind(Channel<T>& channel) {
        if (connected_) {
            throw std::runtime_error("input port is already connected");
        }
        channel_ = &channel;
        reader_id_ = channel.attachReader();
        connected_ = true;
    }

    void ensure() const {
        if (!connected_) {
            throw std::runtime_error("input port is not connected");
        }
    }

    std::string name_;
    Channel<T>* channel_ = nullptr;
    std::size_t reader_id_ = 0;
    bool connected_ = false;
};

} // namespace coropulse
