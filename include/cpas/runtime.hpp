#pragma once

#include "cpas/scheduler.hpp"
#include "cpas/types.hpp"

#include <cstddef>
#include <vector>

namespace cpas {

class TickContext {
public:
    TickContext(TickId tick, Scheduler& scheduler) noexcept
        : tick_(tick), scheduler_(&scheduler) {}

    TickId tick() const noexcept { return tick_; }
    Scheduler& scheduler() const noexcept { return *scheduler_; }

private:
    TickId tick_;
    Scheduler* scheduler_;
};

class Component {
public:
    virtual ~Component() = default;
    virtual Task<void> tick(TickContext& ctx) = 0;
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
          scheduler_(worker_count_) {}

    void setWorkerCount(std::size_t worker_count) {
        worker_count = worker_count == 0 ? 1 : worker_count;
        if (worker_count != worker_count_) {
            throw std::runtime_error(
                "runtime worker count is fixed; construct a new Runtime for a different size");
        }
    }

    void addComponent(Component& component) {
        components_.push_back(&component);
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

        TickContext ctx(tick_, scheduler_);

        for (auto* component : components_) {
            scheduler_.add(component->tick(ctx));
        }

        scheduler_.run();

        for (auto* object : objects_) {
            object->commit(tick_);
        }

        for (auto* object : objects_) {
            object->endTick(tick_);
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
    TickId tick_ = 0;
    std::size_t worker_count_ = 1;
    Scheduler scheduler_;
    std::vector<Component*> components_;
    std::vector<TickObject*> objects_;
};

} // namespace cpas
