#include "sim.hh"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace coropulse;

namespace {

class TimedComponent final : public Component {
public:
    TimedComponent(int id, std::size_t work_rounds,
                   std::vector<std::vector<int>>& order)
        : id_(id), work_rounds_(work_rounds), order_(order) {}

    MAKE_PROCESS({
        const auto tick_index = static_cast<std::size_t>(currentTick() - 1);
        if (tick_index < order_.size()) {
            order_[tick_index].push_back(id_);
        }

        for (std::size_t i = 0; i < work_rounds_; ++i) {
            state_ ^= state_ >> 29;
            state_ *= 0x9e3779b185ebca87ULL;
            state_ ^= state_ >> 33;
            state_ += currentTick() + i;
        }

    })

    std::uint64_t state() const noexcept {
        return state_;
    }

private:
    int id_;
    std::size_t work_rounds_;
    std::vector<std::vector<int>>& order_;
    std::uint64_t state_ = 0x123456789abcdef0ULL;
};

void load_balancer_prioritizes_historical_long_tasks() {
    Simulator sim(1);
    sim.enableLoadBalancing(4);

    std::vector<std::vector<int>> order(2);
    sim.createComponent<TimedComponent>(0, 0, order);
    sim.createComponent<TimedComponent>(1, 0, order);
    auto& heavy = sim.createComponent<TimedComponent>(2, 500000, order);

    sim.tick();
    sim.tick();

    const std::vector<int> first_tick_expected = {0, 1, 2};
    assert(order[0] == first_tick_expected);
    assert(!order[1].empty());
    assert(order[1][0] == 2);
    assert(heavy.state() != 0);
}

} // namespace

int main() {
    load_balancer_prioritizes_historical_long_tasks();
}
