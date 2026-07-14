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

    Task<void> tick(TickContext& ctx) override {
        const auto tick_index = static_cast<std::size_t>(ctx.tick() - 1);
        if (tick_index < order_.size()) {
            order_[tick_index].push_back(id_);
        }

        for (std::size_t i = 0; i < work_rounds_; ++i) {
            state_ ^= state_ >> 29;
            state_ *= 0x9e3779b185ebca87ULL;
            state_ ^= state_ >> 33;
            state_ += ctx.tick() + i;
        }

        co_return;
    }

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
    Runtime runtime(1);
    runtime.enableLoadBalancing(4);

    std::vector<std::vector<int>> order(2);
    TimedComponent light_a(0, 0, order);
    TimedComponent light_b(1, 0, order);
    TimedComponent heavy(2, 500000, order);

    runtime.addComponent(light_a);
    runtime.addComponent(light_b);
    runtime.addComponent(heavy);

    runtime.runTick();
    runtime.runTick();

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
