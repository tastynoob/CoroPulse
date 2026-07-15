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

    MAKE_PROCESS({
        if (!sent_) {
            const bool ok = out.write(42);
            assert(ok);
            sent_ = true;
        }
    })

private:
    bool sent_ = false;
};

class ApiConsumer final : public Component {
public:
    Input<int> in{"in"};

    explicit ApiConsumer(std::vector<std::optional<int>>& reads) : reads_(reads) {}

    MAKE_PROCESS({
        reads_.push_back(in.read());
    })

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

    MAKE_PROCESS({
        events_.push_back("wait-before");
        const auto value = co_await ready.read();
        assert(value.can_accept);
        events_.push_back("wait-after");
    })

private:
    std::vector<std::string>& events_;
};

class ApiSignalSetter final : public Component {
public:
    SignalOutput<Ready> ready{"ready"};

    explicit ApiSignalSetter(std::vector<std::string>& events) : events_(events) {}

    MAKE_PROCESS({
        events_.push_back("set");
        ready.set(Ready{true});
    })

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

class PersistentLocalComponent final : public Component {
public:
    explicit PersistentLocalComponent(std::vector<int>& values) : values_(values) {}

    Task<void> process() override {
        int local_counter = 0;
        for (;;) {
            values_.push_back(++local_counter);
            co_yield tickDone{};
        }
    }

private:
    std::vector<int>& values_;
};

void process_frame_persists_across_ticks() {
    Simulator sim;
    std::vector<int> values;
    sim.createComponent<PersistentLocalComponent>(values);

    sim.run(3);

    const std::vector<int> expected = {1, 2, 3};
    assert(values == expected);
}

class ReturningProcess final : public Component {
public:
    Task<void> process() override {
        co_return;
    }
};

void process_return_is_reported_as_error() {
    Simulator sim;
    sim.createComponent<ReturningProcess>();

    bool threw = false;
    try {
        sim.tick();
    } catch (const std::runtime_error& error) {
        threw = std::string(error.what()).find("component process returned unexpectedly") !=
                std::string::npos;
    }
    assert(threw);
}

} // namespace

int main() {
    simulator_connects_channel_ports();
    simulator_connects_signal_ports();
    simulator_freezes_topology_after_first_tick();
    process_frame_persists_across_ticks();
    process_return_is_reported_as_error();
}
