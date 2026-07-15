#include "sim.hh"

#include <atomic>
#include <cassert>
#include <vector>

using namespace coropulse;

struct Ready {
    bool can_accept = false;
};

struct FanoutProducer final : Component {
    Output<int> out{"out"};
    bool sent = false;

    MAKE_PROCESS({
        if (!sent) {
            const bool ok = out.write(7);
            assert(ok);
            sent = true;
        }
    })
};

struct AtomicChannelConsumer final : Component {
    Input<int> in{"in"};
    std::atomic<int>& empty_reads;
    std::atomic<int>& value_reads;
    std::atomic<int>& value_sum;

    AtomicChannelConsumer(std::atomic<int>& empty_reads, std::atomic<int>& value_reads,
                          std::atomic<int>& value_sum)
        : empty_reads(empty_reads), value_reads(value_reads), value_sum(value_sum) {}

    MAKE_PROCESS({
        auto value = in.read();
        if (value) {
            value_reads.fetch_add(1, std::memory_order_relaxed);
            value_sum.fetch_add(*value, std::memory_order_relaxed);
        } else {
            empty_reads.fetch_add(1, std::memory_order_relaxed);
        }
    })
};

template <class OutputT, class InputT>
void connectMany(Simulator& sim, OutputT& output, const std::vector<InputT*>& inputs) {
    assert(!inputs.empty());
    sim.connect(output, inputs);
}

void channel_fanout_is_safe_with_many_workers() {
    constexpr int kWorkers = 8;
    constexpr int kConsumers = 256;

    Simulator sim(kWorkers);

    std::atomic<int> empty_reads{0};
    std::atomic<int> value_reads{0};
    std::atomic<int> value_sum{0};

    auto& producer = sim.createComponent<FanoutProducer>();
    std::vector<Input<int>*> inputs;
    inputs.reserve(kConsumers);
    for (int i = 0; i < kConsumers; ++i) {
        auto& consumer = sim.createComponent<AtomicChannelConsumer>(
            empty_reads, value_reads, value_sum);
        inputs.push_back(&consumer.in);
    }
    connectMany(sim, producer.out, inputs);

    sim.tick();
    assert(empty_reads.load() == kConsumers);
    assert(value_reads.load() == 0);

    sim.tick();
    assert(empty_reads.load() == kConsumers);
    assert(value_reads.load() == kConsumers);
    assert(value_sum.load() == kConsumers * 7);

    sim.tick();
    assert(empty_reads.load() == kConsumers * 2);
    assert(value_reads.load() == kConsumers);
}

struct SignalWaiter final : Component {
    SignalInput<int> signal{"signal"};
    std::atomic<int>& reads;
    std::atomic<int>& sum;

    SignalWaiter(std::atomic<int>& reads, std::atomic<int>& sum) : reads(reads), sum(sum) {}

    MAKE_PROCESS({
        const int value = co_await signal.read();
        reads.fetch_add(1, std::memory_order_relaxed);
        sum.fetch_add(value, std::memory_order_relaxed);
    })
};

struct SignalSetter final : Component {
    SignalOutput<int> signal{"signal"};

    MAKE_PROCESS({
        signal.set(11);
    })
};

void signal_broadcast_wakes_many_waiters_on_many_workers() {
    constexpr int kWorkers = 8;
    constexpr int kWaiters = 256;

    Simulator sim(kWorkers);

    std::atomic<int> reads{0};
    std::atomic<int> sum{0};

    std::vector<SignalInput<int>*> inputs;
    inputs.reserve(kWaiters);
    for (int i = 0; i < kWaiters; ++i) {
        auto& waiter = sim.createComponent<SignalWaiter>(reads, sum);
        inputs.push_back(&waiter.signal);
    }

    auto& setter = sim.createComponent<SignalSetter>();
    connectMany(sim, setter.signal, inputs);

    sim.tick();
    assert(reads.load() == kWaiters);
    assert(sum.load() == kWaiters * 11);
}

struct ChainUpstream final : Component {
    SignalInput<Ready> ready{"ready"};
    std::atomic<int>& sends;

    explicit ChainUpstream(std::atomic<int>& sends) : sends(sends) {}

    MAKE_PROCESS({
        const auto value = co_await ready.read();
        if (value.can_accept) {
            sends.fetch_add(1, std::memory_order_relaxed);
        }
    })
};

struct ChainMiddle final : Component {
    SignalInput<Ready> sink_ready{"sink-ready"};
    SignalOutput<Ready> upstream_ready{"upstream-ready"};
    std::atomic<int>& forwards;

    explicit ChainMiddle(std::atomic<int>& forwards) : forwards(forwards) {}

    MAKE_PROCESS({
        const auto value = co_await sink_ready.read();
        if (value.can_accept) {
            forwards.fetch_add(1, std::memory_order_relaxed);
            upstream_ready.set(Ready{true});
        }
    })
};

struct ChainSink final : Component {
    SignalOutput<Ready> ready{"ready"};

    MAKE_PROCESS({
        ready.set(Ready{true});
    })
};

void many_backpressure_chains_progress_concurrently() {
    constexpr int kWorkers = 8;
    constexpr int kChains = 128;

    Simulator sim(kWorkers);
    std::atomic<int> sends{0};
    std::atomic<int> forwards{0};

    for (int i = 0; i < kChains; ++i) {
        auto& upstream = sim.createComponent<ChainUpstream>(sends);
        auto& middle = sim.createComponent<ChainMiddle>(forwards);
        auto& sink = sim.createComponent<ChainSink>();
        sim.connect(sink.ready, middle.sink_ready);
        sim.connect(middle.upstream_ready, upstream.ready);
    }

    sim.tick();
    assert(forwards.load() == kChains);
    assert(sends.load() == kChains);
}

int main() {
    channel_fanout_is_safe_with_many_workers();
    signal_broadcast_wakes_many_waiters_on_many_workers();
    many_backpressure_chains_progress_concurrently();
}
