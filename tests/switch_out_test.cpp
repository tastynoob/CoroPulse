#include "sim.hh"

#include <atomic>
#include <cassert>
#include <string>
#include <vector>

using namespace coropulse;

namespace {

class SwitchingOutRecorder final : public Component {
public:
    explicit SwitchingOutRecorder(std::vector<std::string>& events) : events_(events) {}

    MAKE_PROCESS({
        events_.push_back("switch-out-before-" + std::to_string(currentTick()));
        co_await switchOut();
        events_.push_back("switch-out-after-" + std::to_string(currentTick()));
    })

private:
    std::vector<std::string>& events_;
};

class SimpleRecorder final : public Component {
public:
    explicit SimpleRecorder(std::vector<std::string>& events) : events_(events) {}

    MAKE_PROCESS({
        events_.push_back("simple-" + std::to_string(currentTick()));
    })

private:
    std::vector<std::string>& events_;
};

void switch_out_requeues_continuation_after_ready_work() {
    Simulator sim(1);
    std::vector<std::string> events;

    sim.createComponent<SwitchingOutRecorder>(events);
    sim.createComponent<SimpleRecorder>(events);

    sim.tick();

    const std::vector<std::string> expected = {
        "switch-out-before-1",
        "simple-1",
        "switch-out-after-1",
    };
    assert(events == expected);
}

class RepeatedSwitchOut final : public Component {
public:
    RepeatedSwitchOut(std::atomic<int>& steps, int iterations)
        : steps_(steps), iterations_(iterations) {}

    MAKE_PROCESS({
        for (int i = 0; i < iterations_; ++i) {
            steps_.fetch_add(1, std::memory_order_relaxed);
            co_await switchOut();
        }
        steps_.fetch_add(1, std::memory_order_relaxed);
    })

private:
    std::atomic<int>& steps_;
    int iterations_;
};

void repeated_switch_outs_complete_on_many_workers() {
    constexpr int kWorkers = 4;
    constexpr int kComponents = 32;
    constexpr int kIterations = 16;

    Simulator sim(kWorkers);
    std::atomic<int> steps{0};

    for (int i = 0; i < kComponents; ++i) {
        sim.createComponent<RepeatedSwitchOut>(steps, kIterations);
    }

    sim.tick();

    assert(steps.load() == kComponents * (kIterations + 1));
}

} // namespace

int main() {
    switch_out_requeues_continuation_after_ready_work();
    repeated_switch_outs_complete_on_many_workers();
}
