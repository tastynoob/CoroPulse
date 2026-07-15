#include "sim.hh"

#include <cassert>
#include <chrono>
#include <thread>

using namespace coropulse;
using namespace std::chrono_literals;

struct SlowComponent final : Component {
    MAKE_PROCESS({
        std::this_thread::sleep_for(20ms);
    })
};

void idle_ratio_counts_workers_waiting_for_work() {
    Simulator sim(4);
    sim.createComponent<SlowComponent>();

    sim.tick();

    assert(sim.workerIdleRatio() > 0.0);
    assert(sim.workerIdleRatio() <= 1.0);
}

void idle_ratio_is_cumulative() {
    Simulator sim(4);
    sim.createComponent<SlowComponent>();

    sim.tick();
    sim.tick();

    assert(sim.workerIdleRatio() > 0.0);
    assert(sim.workerIdleRatio() <= 1.0);
}

int main() {
    idle_ratio_counts_workers_waiting_for_work();
    idle_ratio_is_cumulative();
}
