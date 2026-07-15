#include "sim.hh"

#include <cassert>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace coropulse;

struct Ready {
    bool can_accept = false;
};

struct OneShotProducer final : Component {
    Output<int> out{"out"};
    bool sent = false;

    MAKE_PROCESS({
        if (!sent) {
            const bool ok = out.write(42);
            assert(ok);
            sent = true;
        }
    })
};

struct RecordingConsumer final : Component {
    Input<int> in{"in"};
    std::vector<std::optional<int>>& reads;

    explicit RecordingConsumer(std::vector<std::optional<int>>& reads) : reads(reads) {}

    MAKE_PROCESS({
        reads.push_back(in.read());
    })
};

void channel_values_are_visible_next_tick() {
    Simulator sim;

    std::vector<std::optional<int>> reads;
    auto& producer = sim.createComponent<OneShotProducer>();
    auto& consumer = sim.createComponent<RecordingConsumer>(reads);
    sim.connect(producer.out, consumer.in);

    sim.tick();
    assert(reads.size() == 1);
    assert(!reads[0].has_value());

    sim.tick();
    assert(reads.size() == 2);
    assert(reads[1].has_value());
    assert(*reads[1] == 42);
}

struct WaitingComponent final : Component {
    SignalInput<bool> ready{"ready"};
    std::vector<std::string>& events;

    explicit WaitingComponent(std::vector<std::string>& events) : events(events) {}

    MAKE_PROCESS({
        events.push_back("waiter-before");
        const bool value = co_await ready.read();
        assert(value);
        events.push_back("waiter-after");
    })
};

struct SetterComponent final : Component {
    SignalOutput<bool> ready{"ready"};
    std::vector<std::string>& events;

    explicit SetterComponent(std::vector<std::string>& events) : events(events) {}

    MAKE_PROCESS({
        events.push_back("setter");
        ready.set(true);
    })
};

void pending_signal_wakes_coroutine_in_same_tick() {
    Simulator sim;

    std::vector<std::string> events;
    auto& waiter = sim.createComponent<WaitingComponent>(events);
    auto& setter = sim.createComponent<SetterComponent>(events);
    sim.connect(setter.ready, waiter.ready);

    sim.tick();

    const std::vector<std::string> expected = {
        "waiter-before",
        "setter",
        "waiter-after",
    };
    assert(events == expected);
}

struct UpstreamCell final : Component {
    SignalInput<Ready> downstream_ready{"downstream-ready"};
    std::vector<std::string>& events;

    explicit UpstreamCell(std::vector<std::string>& events) : events(events) {}

    MAKE_PROCESS({
        events.push_back("upstream-wait");
        const auto ready = co_await downstream_ready.read();
        assert(ready.can_accept);
        events.push_back("upstream-send");
    })
};

struct MiddleCell final : Component {
    SignalInput<Ready> sink_ready{"sink-ready"};
    SignalOutput<Ready> upstream_ready{"upstream-ready"};
    std::vector<std::string>& events;

    explicit MiddleCell(std::vector<std::string>& events) : events(events) {}

    MAKE_PROCESS({
        events.push_back("middle-wait");
        const auto ready = co_await sink_ready.read();
        assert(ready.can_accept);
        events.push_back("middle-ready");
        upstream_ready.set(Ready{true});
    })
};

struct SinkCell final : Component {
    SignalOutput<Ready> sink_ready{"sink-ready"};
    std::vector<std::string>& events;

    explicit SinkCell(std::vector<std::string>& events) : events(events) {}

    MAKE_PROCESS({
        events.push_back("sink-ready");
        sink_ready.set(Ready{true});
    })
};

void timely_backpressure_propagates_through_signal_chain() {
    Simulator sim;

    std::vector<std::string> events;
    auto& upstream = sim.createComponent<UpstreamCell>(events);
    auto& middle = sim.createComponent<MiddleCell>(events);
    auto& sink = sim.createComponent<SinkCell>(events);
    sim.connect(sink.sink_ready, middle.sink_ready);
    sim.connect(middle.upstream_ready, upstream.downstream_ready);

    sim.tick();

    const std::vector<std::string> expected = {
        "upstream-wait",
        "middle-wait",
        "sink-ready",
        "middle-ready",
        "upstream-send",
    };
    assert(events == expected);
}

struct DeadlockedComponent final : Component {
    SignalInput<bool> signal{"signal"};

    MAKE_PROCESS({
        (void)co_await signal.read();
    })
};

struct UnusedSetter final : Component {
    SignalOutput<bool> signal{"signal"};

    MAKE_PROCESS({})
};

void missing_signal_set_is_reported_as_deadlock() {
    Simulator sim;
    auto& component = sim.createComponent<DeadlockedComponent>();
    auto& setter = sim.createComponent<UnusedSetter>();
    sim.connect(setter.signal, component.signal);

    bool threw = false;
    try {
        sim.tick();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
}

int main() {
    channel_values_are_visible_next_tick();
    pending_signal_wakes_coroutine_in_same_tick();
    timely_backpressure_propagates_through_signal_chain();
    missing_signal_set_is_reported_as_deadlock();
}
