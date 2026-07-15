#pragma once

#include "scheduler.hh"
#include "types.hh"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <stdexcept>
#include <utility>
#include <vector>

namespace coropulse {

class TickContext {
public:
    TickContext() = default;

    TickContext(TickId tick, Scheduler& scheduler) noexcept
        : tick_(tick), scheduler_(&scheduler) {}

    void reset(TickId tick, Scheduler& scheduler) noexcept {
        tick_ = tick;
        scheduler_ = &scheduler;
    }

    TickId tick() const noexcept { return tick_; }
    Scheduler& scheduler() const noexcept { return *scheduler_; }

private:
    TickId tick_;
    Scheduler* scheduler_;
};

class SwitchOutAwaiter {
public:
    explicit SwitchOutAwaiter(TickContext& ctx) noexcept : ctx_(&ctx) {}

    bool await_ready() const noexcept { return false; }

    void await_suspend(Scheduler::Handle handle) const {
        ctx_->scheduler().defer(handle);
    }

    void await_resume() const noexcept {}

private:
    TickContext* ctx_;
};

class Component {
public:
    virtual ~Component() = default;
    virtual Task<void> process() = 0;

    TickId currentTick() const {
        return context().tick();
    }

protected:
    SwitchOutAwaiter switchOut() {
        return SwitchOutAwaiter(context());
    }

    TickContext& context() const {
        if (!ctx_) {
            throw std::runtime_error("component tick context is not active");
        }
        return *ctx_;
    }

private:
    friend class Runtime;

    TickContext* ctx_ = nullptr;
};

class TickObject {
public:
    virtual ~TickObject() = default;
    virtual void beginTick(TickId tick) = 0;
    virtual void commit(TickId tick) = 0;
    virtual void endTick(TickId tick) = 0;
};

class Runtime {
public:
    explicit Runtime(std::size_t worker_count = 1)
        : worker_count_(worker_count == 0 ? 1 : worker_count),
          scheduler_(worker_count_) {
        tick_context_.reset(tick_, scheduler_);
    }

    void setWorkerCount(std::size_t worker_count) {
        worker_count = worker_count == 0 ? 1 : worker_count;
        if (worker_count != worker_count_) {
            throw std::runtime_error(
                "runtime worker count is fixed; construct a new Runtime for a different size");
        }
    }

    void enableLoadBalancing(std::size_t window_ticks) {
        if (window_ticks == 0) {
            throw std::runtime_error("load balancing window must be greater than zero");
        }

        load_balance_window_ = window_ticks;
        load_balancing_enabled_ = true;
        trimProfiles();
    }

    void disableLoadBalancing() noexcept {
        load_balancing_enabled_ = false;
    }

    void addComponent(Component& component) {
        const auto component_id = components_.size();
        auto process = component.process();
        components_.push_back(&component);
        profiles_.emplace_back();
        component.ctx_ = &tick_context_;
        scheduler_.addProcess(std::move(process), component_id, &tick_context_);
    }

    void addObject(TickObject& object) {
        objects_.push_back(&object);
    }

    TickId tick() const noexcept { return tick_; }

    void runTick() {
        ++tick_;

        for (auto* object : objects_) {
            object->beginTick(tick_);
        }

        tick_context_.reset(tick_, scheduler_);

        rebuildComponentOrder();
        scheduler_.run(component_order_, load_balancing_enabled_);
        auto samples = scheduler_.takeSamples();

        for (auto* object : objects_) {
            object->commit(tick_);
        }

        for (auto* object : objects_) {
            object->endTick(tick_);
        }

        if (load_balancing_enabled_) {
            recordSamples(samples);
        }
    }

    void runTick(std::size_t worker_count) {
        worker_count = worker_count == 0 ? 1 : worker_count;
        if (worker_count != worker_count_) {
            throw std::runtime_error(
                "runtime worker count is fixed; construct a new Runtime for a different size");
        }
        runTick();
    }

private:
    struct ComponentProfile {
        std::deque<Scheduler::Duration> samples;
        Scheduler::Duration total{0};
    };

    Scheduler::Duration averageActiveTime(std::size_t component_id) const {
        const auto& profile = profiles_[component_id];
        if (profile.samples.empty()) {
            return Scheduler::Duration{0};
        }
        return profile.total /
               static_cast<Scheduler::Duration::rep>(profile.samples.size());
    }

    void rebuildComponentOrder() {
        component_order_.clear();
        component_order_.reserve(components_.size());
        for (std::size_t i = 0; i < components_.size(); ++i) {
            component_order_.push_back(i);
        }

        if (!load_balancing_enabled_) {
            return;
        }

        std::sort(component_order_.begin(), component_order_.end(),
                  [this](std::size_t lhs, std::size_t rhs) {
                      const auto lhs_avg = averageActiveTime(lhs);
                      const auto rhs_avg = averageActiveTime(rhs);
                      if (lhs_avg != rhs_avg) {
                          return lhs_avg > rhs_avg;
                      }
                      return lhs < rhs;
                  });
    }

    void recordSamples(const std::vector<Scheduler::TaskSample>& samples) {
        for (const auto& sample : samples) {
            if (sample.component_id >= profiles_.size()) {
                continue;
            }

            auto& profile = profiles_[sample.component_id];
            while (profile.samples.size() >= load_balance_window_) {
                profile.total -= profile.samples.front();
                profile.samples.pop_front();
            }

            profile.samples.push_back(sample.active_time);
            profile.total += sample.active_time;
        }
    }

    void trimProfiles() {
        for (auto& profile : profiles_) {
            while (profile.samples.size() > load_balance_window_) {
                profile.total -= profile.samples.front();
                profile.samples.pop_front();
            }
        }
    }

    TickId tick_ = 0;
    std::size_t worker_count_ = 1;
    Scheduler scheduler_;
    TickContext tick_context_;
    std::vector<Component*> components_;
    std::vector<TickObject*> objects_;
    std::vector<ComponentProfile> profiles_;
    std::vector<std::size_t> component_order_;
    std::size_t load_balance_window_ = 0;
    bool load_balancing_enabled_ = false;
};

} // namespace coropulse

#define MAKE_PROCESS(...)                                      \
    ::coropulse::Task<void> process() override {               \
        for (;; co_yield ::coropulse::tickDone{}) __VA_ARGS__  \
        co_return;                                             \
    }
