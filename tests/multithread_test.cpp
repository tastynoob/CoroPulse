#include "cpas/sim.hpp"

#include <atomic>
#include <cassert>
#include <memory>
#include <vector>

using namespace cpas;

struct Ready {
    bool can_accept = false;
};

struct FanoutProducer final : Component {
    Channel<int>::Master out;
    bool sent = false;

    explicit FanoutProducer(Channel<int>::Master out) : out(out) {}

    Task<void> tick(TickContext& ctx) override {
        if (!sent) {
            const bool ok = out.write(ctx, 7);
            assert(ok);
            sent = true;
        }
        co_return;
    }
};

struct AtomicChannelConsumer final : Component {
    Channel<int>::Slaver in;
    std::atomic<int>& empty_reads;
    std::atomic<int>& value_reads;
    std::atomic<int>& value_sum;

    AtomicChannelConsumer(Channel<int>::Slaver in, std::atomic<int>& empty_reads,
                          std::atomic<int>& value_reads, std::atomic<int>& value_sum)
        : in(in), empty_reads(empty_reads), value_reads(value_reads), value_sum(value_sum) {}

    Task<void> tick(TickContext& ctx) override {
        auto value = in.read(ctx);
        if (value) {
            value_reads.fetch_add(1, std::memory_order_relaxed);
            value_sum.fetch_add(*value, std::memory_order_relaxed);
        } else {
            empty_reads.fetch_add(1, std::memory_order_relaxed);
        }
        co_return;
    }
};

void channel_fanout_is_safe_with_many_workers() {
    constexpr int kWorkers = 8;
    constexpr int kConsumers = 256;

    Runtime runtime(kWorkers);
    Channel<int> channel("fanout");
    runtime.addObject(channel);

    std::atomic<int> empty_reads{0};
    std::atomic<int> value_reads{0};
    std::atomic<int> value_sum{0};

    FanoutProducer producer(channel.master());
    runtime.addComponent(producer);

    std::vector<std::unique_ptr<AtomicChannelConsumer>> consumers;
    consumers.reserve(kConsumers);
    for (int i = 0; i < kConsumers; ++i) {
        consumers.push_back(std::make_unique<AtomicChannelConsumer>(
            channel.addSlaver(), empty_reads, value_reads, value_sum));
        runtime.addComponent(*consumers.back());
    }

    runtime.runTick();
    assert(empty_reads.load() == kConsumers);
    assert(value_reads.load() == 0);

    runtime.runTick();
    assert(empty_reads.load() == kConsumers);
    assert(value_reads.load() == kConsumers);
    assert(value_sum.load() == kConsumers * 7);

    runtime.runTick();
    assert(empty_reads.load() == kConsumers * 2);
    assert(value_reads.load() == kConsumers);
}

struct SignalWaiter final : Component {
    Signal<int>::Slaver signal;
    std::atomic<int>& reads;
    std::atomic<int>& sum;

    SignalWaiter(Signal<int>::Slaver signal, std::atomic<int>& reads, std::atomic<int>& sum)
        : signal(signal), reads(reads), sum(sum) {}

    Task<void> tick(TickContext& ctx) override {
        const int value = co_await signal.read(ctx);
        reads.fetch_add(1, std::memory_order_relaxed);
        sum.fetch_add(value, std::memory_order_relaxed);
        co_return;
    }
};

struct SignalSetter final : Component {
    Signal<int>::Master signal;

    explicit SignalSetter(Signal<int>::Master signal) : signal(signal) {}

    Task<void> tick(TickContext& ctx) override {
        signal.set(ctx, 11);
        co_return;
    }
};

void signal_broadcast_wakes_many_waiters_on_many_workers() {
    constexpr int kWorkers = 8;
    constexpr int kWaiters = 256;

    Runtime runtime(kWorkers);
    Signal<int> signal("broadcast-ready");
    runtime.addObject(signal);

    std::atomic<int> reads{0};
    std::atomic<int> sum{0};

    std::vector<std::unique_ptr<SignalWaiter>> waiters;
    waiters.reserve(kWaiters);
    for (int i = 0; i < kWaiters; ++i) {
        waiters.push_back(std::make_unique<SignalWaiter>(signal.addSlaver(), reads, sum));
        runtime.addComponent(*waiters.back());
    }

    SignalSetter setter(signal.master());
    runtime.addComponent(setter);

    runtime.runTick();
    assert(reads.load() == kWaiters);
    assert(sum.load() == kWaiters * 11);
}

struct ChainUpstream final : Component {
    Signal<Ready>::Slaver ready;
    std::atomic<int>& sends;

    ChainUpstream(Signal<Ready>::Slaver ready, std::atomic<int>& sends)
        : ready(ready), sends(sends) {}

    Task<void> tick(TickContext& ctx) override {
        const auto value = co_await ready.read(ctx);
        if (value.can_accept) {
            sends.fetch_add(1, std::memory_order_relaxed);
        }
        co_return;
    }
};

struct ChainMiddle final : Component {
    Signal<Ready>::Slaver sink_ready;
    Signal<Ready>::Master upstream_ready;
    std::atomic<int>& forwards;

    ChainMiddle(Signal<Ready>::Slaver sink_ready, Signal<Ready>::Master upstream_ready,
                std::atomic<int>& forwards)
        : sink_ready(sink_ready), upstream_ready(upstream_ready), forwards(forwards) {}

    Task<void> tick(TickContext& ctx) override {
        const auto value = co_await sink_ready.read(ctx);
        if (value.can_accept) {
            forwards.fetch_add(1, std::memory_order_relaxed);
            upstream_ready.set(ctx, Ready{true});
        }
        co_return;
    }
};

struct ChainSink final : Component {
    Signal<Ready>::Master ready;

    explicit ChainSink(Signal<Ready>::Master ready) : ready(ready) {}

    Task<void> tick(TickContext& ctx) override {
        ready.set(ctx, Ready{true});
        co_return;
    }
};

void many_backpressure_chains_progress_concurrently() {
    constexpr int kWorkers = 8;
    constexpr int kChains = 128;

    Runtime runtime(kWorkers);
    std::atomic<int> sends{0};
    std::atomic<int> forwards{0};

    std::vector<std::unique_ptr<Signal<Ready>>> sink_ready;
    std::vector<std::unique_ptr<Signal<Ready>>> upstream_ready;
    std::vector<std::unique_ptr<ChainUpstream>> upstreams;
    std::vector<std::unique_ptr<ChainMiddle>> middles;
    std::vector<std::unique_ptr<ChainSink>> sinks;

    sink_ready.reserve(kChains);
    upstream_ready.reserve(kChains);
    upstreams.reserve(kChains);
    middles.reserve(kChains);
    sinks.reserve(kChains);

    for (int i = 0; i < kChains; ++i) {
        sink_ready.push_back(std::make_unique<Signal<Ready>>("chain-sink"));
        upstream_ready.push_back(std::make_unique<Signal<Ready>>("chain-upstream"));
        runtime.addObject(*sink_ready.back());
        runtime.addObject(*upstream_ready.back());

        upstreams.push_back(
            std::make_unique<ChainUpstream>(upstream_ready.back()->addSlaver(), sends));
        middles.push_back(std::make_unique<ChainMiddle>(
            sink_ready.back()->addSlaver(), upstream_ready.back()->master(), forwards));
        sinks.push_back(std::make_unique<ChainSink>(sink_ready.back()->master()));

        runtime.addComponent(*upstreams.back());
        runtime.addComponent(*middles.back());
        runtime.addComponent(*sinks.back());
    }

    runtime.runTick();
    assert(forwards.load() == kChains);
    assert(sends.load() == kChains);
}

int main() {
    channel_fanout_is_safe_with_many_workers();
    signal_broadcast_wakes_many_waiters_on_many_workers();
    many_backpressure_chains_progress_concurrently();
}
