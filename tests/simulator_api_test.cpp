#include "sim.hh"

#include <cassert>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace coropulse;

namespace {

struct Ready {
    bool can_accept = false;
};

class ApiProducer final : public Component {
public:
    Output<int> out{"out"};

    Task<void> tick() override {
        if (!sent_) {
            const bool ok = out.write(42);
            assert(ok);
            sent_ = true;
        }
        co_return;
    }

private:
    bool sent_ = false;
};

class ApiConsumer final : public Component {
public:
    Input<int> in{"in"};

    explicit ApiConsumer(std::vector<std::optional<int>>& reads) : reads_(reads) {}

    Task<void> tick() override {
        reads_.push_back(in.read());
        co_return;
    }

private:
    std::vector<std::optional<int>>& reads_;
};

void simulator_connects_channel_ports() {
    Simulator sim;

    std::vector<std::optional<int>> first_reads;
    std::vector<std::optional<int>> second_reads;
    auto& producer = sim.createComponent<ApiProducer>();
    auto& first = sim.createComponent<ApiConsumer>(first_reads);
    auto& second = sim.createComponent<ApiConsumer>(second_reads);

    sim.connect(producer.out, first.in, second.in);

    sim.tick();
    assert(first_reads.size() == 1);
    assert(second_reads.size() == 1);
    assert(!first_reads[0].has_value());
    assert(!second_reads[0].has_value());

    sim.tick();
    assert(first_reads.size() == 2);
    assert(second_reads.size() == 2);
    assert(first_reads[1].has_value());
    assert(second_reads[1].has_value());
    assert(*first_reads[1] == 42);
    assert(*second_reads[1] == 42);
}

class ApiWaitingComponent final : public Component {
public:
    SignalInput<Ready> ready{"ready"};

    explicit ApiWaitingComponent(std::vector<std::string>& events) : events_(events) {}

    Task<void> tick() override {
        events_.push_back("wait-before");
        const auto value = co_await ready.read();
        assert(value.can_accept);
        events_.push_back("wait-after");
        co_return;
    }

private:
    std::vector<std::string>& events_;
};

class ApiSignalSetter final : public Component {
public:
    SignalOutput<Ready> ready{"ready"};

    explicit ApiSignalSetter(std::vector<std::string>& events) : events_(events) {}

    Task<void> tick() override {
        events_.push_back("set");
        ready.set(Ready{true});
        co_return;
    }

private:
    std::vector<std::string>& events_;
};

void simulator_connects_signal_ports() {
    Simulator sim;
    std::vector<std::string> events;

    auto& waiter = sim.createComponent<ApiWaitingComponent>(events);
    auto& setter = sim.createComponent<ApiSignalSetter>(events);

    sim.connect(setter.ready, waiter.ready);
    sim.tick();

    const std::vector<std::string> expected = {
        "wait-before",
        "set",
        "wait-after",
    };
    assert(events == expected);
}

void simulator_freezes_topology_after_first_tick() {
    Simulator sim;
    std::vector<std::optional<int>> reads;
    auto& producer = sim.createComponent<ApiProducer>();
    auto& consumer = sim.createComponent<ApiConsumer>(reads);
    sim.connect(producer.out, consumer.in);

    sim.tick();

    bool threw = false;
    try {
        (void)sim.createComponent<ApiProducer>();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
}

} // namespace

int main() {
    simulator_connects_channel_ports();
    simulator_connects_signal_ports();
    simulator_freezes_topology_after_first_tick();
}
