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
    Channel<int>::Master out;
    bool sent = false;

    explicit OneShotProducer(Channel<int>::Master out) : out(out) {}

    Task<void> tick(TickContext& ctx) override {
        if (!sent) {
            const bool ok = out.write(ctx, 42);
            assert(ok);
            sent = true;
        }
        co_return;
    }
};

struct RecordingConsumer final : Component {
    Channel<int>::Slaver in;
    std::vector<std::optional<int>>& reads;

    RecordingConsumer(Channel<int>::Slaver in, std::vector<std::optional<int>>& reads)
        : in(in), reads(reads) {}

    Task<void> tick(TickContext& ctx) override {
        reads.push_back(in.read(ctx));
        co_return;
    }
};

void channel_values_are_visible_next_tick() {
    Runtime runtime;
    Channel<int> channel("numbers");
    runtime.addObject(channel);

    std::vector<std::optional<int>> reads;
    RecordingConsumer consumer(channel.addSlaver(), reads);
    OneShotProducer producer(channel.master());

    runtime.addComponent(consumer);
    runtime.addComponent(producer);

    runtime.runTick();
    assert(reads.size() == 1);
    assert(!reads[0].has_value());

    runtime.runTick();
    assert(reads.size() == 2);
    assert(reads[1].has_value());
    assert(*reads[1] == 42);
}

struct WaitingComponent final : Component {
    Signal<bool>::Slaver ready;
    std::vector<std::string>& events;

    WaitingComponent(Signal<bool>::Slaver ready, std::vector<std::string>& events)
        : ready(ready), events(events) {}

    Task<void> tick(TickContext& ctx) override {
        events.push_back("waiter-before");
        const bool value = co_await ready.read(ctx);
        assert(value);
        events.push_back("waiter-after");
        co_return;
    }
};

struct SetterComponent final : Component {
    Signal<bool>::Master ready;
    std::vector<std::string>& events;

    SetterComponent(Signal<bool>::Master ready, std::vector<std::string>& events)
        : ready(ready), events(events) {}

    Task<void> tick(TickContext& ctx) override {
        events.push_back("setter");
        ready.set(ctx, true);
        co_return;
    }
};

void pending_signal_wakes_coroutine_in_same_tick() {
    Runtime runtime;
    Signal<bool> signal("ready");
    runtime.addObject(signal);

    std::vector<std::string> events;
    WaitingComponent waiter(signal.addSlaver(), events);
    SetterComponent setter(signal.master(), events);

    runtime.addComponent(waiter);
    runtime.addComponent(setter);

    runtime.runTick();

    const std::vector<std::string> expected = {
        "waiter-before",
        "setter",
        "waiter-after",
    };
    assert(events == expected);
}

struct UpstreamCell final : Component {
    Signal<Ready>::Slaver downstream_ready;
    std::vector<std::string>& events;

    UpstreamCell(Signal<Ready>::Slaver downstream_ready, std::vector<std::string>& events)
        : downstream_ready(downstream_ready), events(events) {}

    Task<void> tick(TickContext& ctx) override {
        events.push_back("upstream-wait");
        const auto ready = co_await downstream_ready.read(ctx);
        assert(ready.can_accept);
        events.push_back("upstream-send");
        co_return;
    }
};

struct MiddleCell final : Component {
    Signal<Ready>::Slaver sink_ready;
    Signal<Ready>::Master upstream_ready;
    std::vector<std::string>& events;

    MiddleCell(Signal<Ready>::Slaver sink_ready, Signal<Ready>::Master upstream_ready,
               std::vector<std::string>& events)
        : sink_ready(sink_ready), upstream_ready(upstream_ready), events(events) {}

    Task<void> tick(TickContext& ctx) override {
        events.push_back("middle-wait");
        const auto ready = co_await sink_ready.read(ctx);
        assert(ready.can_accept);
        events.push_back("middle-ready");
        upstream_ready.set(ctx, Ready{true});
        co_return;
    }
};

struct SinkCell final : Component {
    Signal<Ready>::Master sink_ready;
    std::vector<std::string>& events;

    SinkCell(Signal<Ready>::Master sink_ready, std::vector<std::string>& events)
        : sink_ready(sink_ready), events(events) {}

    Task<void> tick(TickContext& ctx) override {
        events.push_back("sink-ready");
        sink_ready.set(ctx, Ready{true});
        co_return;
    }
};

void timely_backpressure_propagates_through_signal_chain() {
    Runtime runtime;
    Signal<Ready> sink_ready("sink-ready");
    Signal<Ready> upstream_ready("upstream-ready");
    runtime.addObject(sink_ready);
    runtime.addObject(upstream_ready);

    std::vector<std::string> events;
    UpstreamCell upstream(upstream_ready.addSlaver(), events);
    MiddleCell middle(sink_ready.addSlaver(), upstream_ready.master(), events);
    SinkCell sink(sink_ready.master(), events);

    runtime.addComponent(upstream);
    runtime.addComponent(middle);
    runtime.addComponent(sink);

    runtime.runTick();

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
    Signal<bool>::Slaver signal;

    explicit DeadlockedComponent(Signal<bool>::Slaver signal) : signal(signal) {}

    Task<void> tick(TickContext& ctx) override {
        (void)co_await signal.read(ctx);
        co_return;
    }
};

void missing_signal_set_is_reported_as_deadlock() {
    Runtime runtime;
    Signal<bool> signal("never-set");
    runtime.addObject(signal);

    DeadlockedComponent component(signal.addSlaver());
    runtime.addComponent(component);

    bool threw = false;
    try {
        runtime.runTick();
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
