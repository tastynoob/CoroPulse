#include "sim.hh"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <vector>

using namespace coropulse;

namespace {

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

class StressModule final : public Component {
public:
    Input<std::uint64_t> in{"in"};
    Output<std::uint64_t> out{"out"};

    StressModule(std::uint64_t stage_id, std::uint64_t work_rounds,
                 bool has_input, bool has_output)
        : stage_id_(stage_id),
          work_rounds_(work_rounds),
          has_input_(has_input),
          has_output_(has_output),
          state_(0x123456789abcdef0ULL ^ (stage_id * 0x100000001b3ULL)) {}

    MAKE_PROCESS({
        if (has_input_) {
            auto value = in.read();
            if (value) {
                state_ ^= *value + stage_id_ * 0x517cc1b727220a95ULL;
                pending_output_ = state_ ^ (*value << 1);
                ++accepted_;
            }
        } else {
            pending_output_ = state_ + produced_ + stage_id_;
            ++produced_;
        }

        state_ = burnCpu(state_ + currentTick() + stage_id_, work_rounds_);

        if (has_output_ && pending_output_) {
            const auto value = *pending_output_ ^ state_;
            if (out.write(value)) {
                pending_output_.reset();
            }
        } else if (!has_output_) {
            checksum_ ^= state_ + pending_output_.value_or(0);
            pending_output_.reset();
        }

    })

    std::uint64_t checksum() const {
        return checksum_ ^ state_ ^ (accepted_ << 32) ^ produced_;
    }

private:
    std::uint64_t stage_id_;
    std::uint64_t work_rounds_;
    bool has_input_;
    bool has_output_;
    std::optional<std::uint64_t> pending_output_;
    std::uint64_t state_;
    std::uint64_t checksum_ = 0;
    std::uint64_t accepted_ = 0;
    std::uint64_t produced_ = 0;
};

struct BenchResult {
    std::size_t workers;
    std::uint64_t ticks;
    std::uint64_t work_rounds;
    double seconds;
    double ticks_per_second;
    double idle_ratio;
    std::uint64_t checksum;
};

BenchResult runBench(std::size_t workers, std::uint64_t ticks, std::uint64_t work_rounds) {
    Simulator sim(workers);

    auto& a = sim.createComponent<StressModule>(1, work_rounds, false, true);
    auto& b = sim.createComponent<StressModule>(2, work_rounds, true, true);
    auto& c = sim.createComponent<StressModule>(3, work_rounds, true, true);
    auto& d = sim.createComponent<StressModule>(4, work_rounds, true, false);
    sim.connect(a.out, b.in);
    sim.connect(b.out, c.in);
    sim.connect(c.out, d.in);

    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < ticks; ++i) {
        sim.tick();
    }
    const auto end = std::chrono::steady_clock::now();

    const auto seconds = std::chrono::duration<double>(end - start).count();
    return BenchResult{
        workers,
        ticks,
        work_rounds,
        seconds,
        static_cast<double>(ticks) / seconds,
        sim.workerIdleRatio(),
        a.checksum() ^ b.checksum() ^ c.checksum() ^ d.checksum(),
    };
}

void printResult(const BenchResult& result, double baseline_seconds) {
    const double speedup = baseline_seconds / result.seconds;
    std::cout << std::setw(7) << result.workers
              << std::setw(14) << std::fixed << std::setprecision(4) << result.seconds
              << std::setw(16) << std::fixed << std::setprecision(1) << result.ticks_per_second
              << std::setw(12) << std::fixed << std::setprecision(2) << speedup
              << std::setw(11) << std::fixed << std::setprecision(1)
              << (result.idle_ratio * 100.0)
              << "    0x" << std::hex << result.checksum << std::dec << '\n';
}

} // namespace

int main(int argc, char** argv) {
    const auto ticks = parseArg(argv, argc, 1, 100);
    const auto work_rounds = parseArg(argv, argc, 2, 3000000);
    const std::vector<std::size_t> worker_counts = {1, 2, 4};

    std::cout << "4-module A->B->C->D rate benchmark\n";
    std::cout << "ticks=" << ticks << ", work_rounds_per_module_per_tick=" << work_rounds
              << '\n';
    std::cout << "usage: ./rate_bench [ticks] [work_rounds_per_module_per_tick]\n";
    std::cout << std::setw(7) << "workers"
              << std::setw(14) << "seconds"
              << std::setw(16) << "ticks/s"
              << std::setw(12) << "speedup"
              << std::setw(11) << "idle%"
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
