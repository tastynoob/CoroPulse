#include "sim.hh"

#include <atomic>
#include <cassert>
#include <string>
#include <vector>

using namespace coropulse;

namespace {

class YieldingRecorder final : public Component {
public:
    explicit YieldingRecorder(std::vector<std::string>& events) : events_(events) {}

    Task<void> tick() override {
        events_.push_back("yield-before-" + std::to_string(currentTick()));
        co_await yield();
        events_.push_back("yield-after-" + std::to_string(currentTick()));
        co_return;
    }

private:
    std::vector<std::string>& events_;
};

class SimpleRecorder final : public Component {
public:
    explicit SimpleRecorder(std::vector<std::string>& events) : events_(events) {}

    Task<void> tick() override {
        events_.push_back("simple-" + std::to_string(currentTick()));
        co_return;
    }

private:
    std::vector<std::string>& events_;
};

void yield_requeues_continuation_after_ready_work() {
    Simulator sim(1);
    std::vector<std::string> events;

    sim.createComponent<YieldingRecorder>(events);
    sim.createComponent<SimpleRecorder>(events);

    sim.tick();

    const std::vector<std::string> expected = {
        "yield-before-1",
        "simple-1",
        "yield-after-1",
    };
    assert(events == expected);
}

class RepeatedYield final : public Component {
public:
    RepeatedYield(std::atomic<int>& steps, int iterations)
        : steps_(steps), iterations_(iterations) {}

    Task<void> tick() override {
        for (int i = 0; i < iterations_; ++i) {
            steps_.fetch_add(1, std::memory_order_relaxed);
            co_await yield();
        }
        steps_.fetch_add(1, std::memory_order_relaxed);
        co_return;
    }

private:
    std::atomic<int>& steps_;
    int iterations_;
};

void repeated_yields_complete_on_many_workers() {
    constexpr int kWorkers = 4;
    constexpr int kComponents = 32;
    constexpr int kIterations = 16;

    Simulator sim(kWorkers);
    std::atomic<int> steps{0};

    for (int i = 0; i < kComponents; ++i) {
        sim.createComponent<RepeatedYield>(steps, kIterations);
    }

    sim.tick();

    assert(steps.load() == kComponents * (kIterations + 1));
}

} // namespace

int main() {
    yield_requeues_continuation_after_ready_work();
    repeated_yields_complete_on_many_workers();
}
