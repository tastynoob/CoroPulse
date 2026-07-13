#include "cpas/sim.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string_view>
#include <vector>

using namespace cpas;

namespace {

struct Ready {
    bool can_accept = false;
};

std::uint64_t parseArg(char** argv, int argc, int index, std::uint64_t fallback) {
    if (index >= argc) {
        return fallback;
    }
    char* end = nullptr;
    const auto value = std::strtoull(argv[index], &end, 10);
    if (end == argv[index] || *end != '\0' || value == 0) {
        return fallback;
    }
    return value;
}

std::uint64_t burnCpu(std::uint64_t value, std::uint64_t rounds) {
    for (std::uint64_t i = 0; i < rounds; ++i) {
        value ^= value >> 29;
        value *= 0x9e3779b185ebca87ULL;
        value ^= value >> 33;
        value += 0x94d049bb133111ebULL + i;
    }
    return value;
}

bool coinFlip(std::uint64_t tick, std::uint64_t stage_id) {
    auto x = tick + stage_id * 0x9e3779b97f4a7c15ULL;
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return (x & 1ULL) != 0;
}

class SignalStressModule final : public Component {
public:
    SignalStressModule(std::string_view name, std::uint64_t stage_id,
                       std::uint64_t work_rounds,
                       std::optional<Signal<Ready>::Slaver> downstream_ready,
                       std::optional<Signal<Ready>::Master> upstream_ready,
                       std::optional<Channel<std::uint64_t>::Slaver> input,
                       std::optional<Channel<std::uint64_t>::Master> output)
        : stage_id_(stage_id),
          work_rounds_(work_rounds),
          downstream_ready_(downstream_ready),
          upstream_ready_(upstream_ready),
          input_(input),
          output_(output),
          state_(0x6a09e667f3bcc909ULL ^ (stage_id * 0x100000001b3ULL)) {
        (void)name;
    }

    Task<void> tick(TickContext& ctx) override {
        bool local_accept = true;
        if (upstream_ready_) {
            local_accept = coinFlip(ctx.tick(), stage_id_);
            if (local_accept) {
                ++ready_true_;
            } else {
                ++ready_false_;
            }
            upstream_ready_->set(ctx, Ready{local_accept});
        }

        bool downstream_accept = true;
        if (downstream_ready_) {
            const auto ready = co_await downstream_ready_->read(ctx);
            downstream_accept = ready.can_accept;
        }

        std::optional<std::uint64_t> input_value;
        if (input_ && local_accept) {
            input_value = input_->read(ctx);
            if (input_value) {
                ++accepted_input_;
                state_ ^= *input_value + stage_id_ * 0x517cc1b727220a95ULL;
            }
        }

        const bool has_payload = !input_ || input_value.has_value();
        if (downstream_accept && has_payload) {
            ++sent_or_progressed_;
            state_ ^= ctx.tick() * 0x517cc1b727220a95ULL;
        } else {
            ++blocked_;
            state_ ^= ctx.tick() * 0x94d049bb133111ebULL;
        }

        state_ = burnCpu(state_ + stage_id_ + sent_or_progressed_, work_rounds_);
        checksum_ ^= state_ + (sent_or_progressed_ << 7) + (blocked_ << 17);

        if (output_ && downstream_accept && has_payload) {
            const auto value = state_ ^ input_value.value_or(ctx.tick() + stage_id_);
            if (!output_->write(ctx, value)) {
                ++blocked_;
            }
        }

        co_return;
    }

    std::uint64_t checksum() const {
        return checksum_ ^ state_ ^ (ready_true_ << 8) ^ (ready_false_ << 24) ^
               (sent_or_progressed_ << 40) ^ (accepted_input_ << 48) ^ blocked_;
    }

private:
    std::uint64_t stage_id_;
    std::uint64_t work_rounds_;
    std::optional<Signal<Ready>::Slaver> downstream_ready_;
    std::optional<Signal<Ready>::Master> upstream_ready_;
    std::optional<Channel<std::uint64_t>::Slaver> input_;
    std::optional<Channel<std::uint64_t>::Master> output_;
    std::uint64_t state_;
    std::uint64_t checksum_ = 0;
    std::uint64_t ready_true_ = 0;
    std::uint64_t ready_false_ = 0;
    std::uint64_t accepted_input_ = 0;
    std::uint64_t sent_or_progressed_ = 0;
    std::uint64_t blocked_ = 0;
};

struct BenchResult {
    std::size_t workers;
    std::uint64_t ticks;
    std::uint64_t work_rounds;
    double seconds;
    double ticks_per_second;
    std::uint64_t checksum;
};

BenchResult runBench(std::size_t workers, std::uint64_t ticks, std::uint64_t work_rounds) {
    Runtime runtime(workers);

    Signal<Ready> b_ready("B.ready -> A");
    Signal<Ready> d_ready("D.ready -> C");
    Channel<std::uint64_t> ab("A->B");
    Channel<std::uint64_t> bc("B->C");
    Channel<std::uint64_t> cd("C->D");

    runtime.addObject(b_ready);
    runtime.addObject(d_ready);
    runtime.addObject(ab);
    runtime.addObject(bc);
    runtime.addObject(cd);

    SignalStressModule a("A", 1, work_rounds, b_ready.addSlaver(), std::nullopt,
                         std::nullopt, ab.master());
    SignalStressModule b("B", 2, work_rounds, std::nullopt, b_ready.master(),
                         ab.addSlaver(), bc.master());
    SignalStressModule c("C", 3, work_rounds, d_ready.addSlaver(), std::nullopt,
                         bc.addSlaver(), cd.master());
    SignalStressModule d("D", 4, work_rounds, std::nullopt, d_ready.master(),
                         cd.addSlaver(), std::nullopt);

    runtime.addComponent(a);
    runtime.addComponent(b);
    runtime.addComponent(c);
    runtime.addComponent(d);

    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < ticks; ++i) {
        runtime.runTick();
    }
    const auto end = std::chrono::steady_clock::now();

    const auto seconds = std::chrono::duration<double>(end - start).count();
    return BenchResult{
        workers,
        ticks,
        work_rounds,
        seconds,
        static_cast<double>(ticks) / seconds,
        a.checksum() ^ b.checksum() ^ c.checksum() ^ d.checksum(),
    };
}

void printResult(const BenchResult& result, double baseline_seconds) {
    const double speedup = baseline_seconds / result.seconds;
    std::cout << std::setw(7) << result.workers
              << std::setw(14) << std::fixed << std::setprecision(4) << result.seconds
              << std::setw(16) << std::fixed << std::setprecision(1) << result.ticks_per_second
              << std::setw(12) << std::fixed << std::setprecision(2) << speedup
              << "    0x" << std::hex << result.checksum << std::dec << '\n';
}

} // namespace

int main(int argc, char** argv) {
    const auto ticks = parseArg(argv, argc, 1, 100);
    const auto work_rounds = parseArg(argv, argc, 2, 3000000);
    const std::vector<std::size_t> worker_counts = {1, 2, 4};

    std::cout << "4-module A->B->C->D signal backpressure benchmark\n";
    std::cout << "B backpressures A with 50% deterministic probability; "
                 "D backpressures C with 50% deterministic probability\n";
    std::cout << "ticks=" << ticks << ", work_rounds_per_module_per_tick=" << work_rounds
              << '\n';
    std::cout << "usage: ./signal_rate_bench [ticks] [work_rounds_per_module_per_tick]\n";
    std::cout << std::setw(7) << "workers"
              << std::setw(14) << "seconds"
              << std::setw(16) << "ticks/s"
              << std::setw(12) << "speedup"
              << "    checksum\n";

    double baseline_seconds = 0.0;
    for (const auto workers : worker_counts) {
        const auto result = runBench(workers, ticks, work_rounds);
        if (baseline_seconds == 0.0) {
            baseline_seconds = result.seconds;
        }
        printResult(result, baseline_seconds);
    }
}
