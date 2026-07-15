#include "sim.hh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

using namespace coropulse;

namespace {

using Clock = std::chrono::steady_clock;

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

class TickProbe {
public:
    explicit TickProbe(std::size_t light_count)
        : light_done_ns_(light_count) {}

    void beginTick() {
        start_ = Clock::now();
        for (auto& value : light_done_ns_) {
            value.store(0, std::memory_order_relaxed);
        }
    }

    void recordLight(std::size_t index) {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_)
                .count();
        light_done_ns_[index].store(elapsed, std::memory_order_relaxed);
    }

    double maxLightMs() const {
        std::int64_t max_ns = 0;
        for (const auto& value : light_done_ns_) {
            max_ns = std::max(max_ns, value.load(std::memory_order_relaxed));
        }
        return static_cast<double>(max_ns) / 1'000'000.0;
    }

private:
    Clock::time_point start_{};
    std::vector<std::atomic<std::int64_t>> light_done_ns_;
};

class HeavyNoSwitchOut final : public Component {
public:
    HeavyNoSwitchOut(std::uint64_t rounds, std::uint64_t chunks)
        : rounds_(rounds),
          chunks_(chunks),
          state_(0x6a09e667f3bcc909ULL) {}

    MAKE_PROCESS({
        for (std::uint64_t i = 0; i < chunks_; ++i) {
            state_ = burnCpu(state_ + currentTick() + i, rounds_);
        }
        checksum_ ^= state_ + (currentTick() << 7);
    })

    std::uint64_t checksum() const {
        return checksum_ ^ state_;
    }

private:
    std::uint64_t rounds_;
    std::uint64_t chunks_;
    std::uint64_t state_;
    std::uint64_t checksum_ = 0;
};

class HeavySwitchingOut final : public Component {
public:
    HeavySwitchingOut(std::uint64_t rounds, std::uint64_t chunks)
        : rounds_(rounds),
          chunks_(chunks),
          state_(0x6a09e667f3bcc909ULL) {}

    MAKE_PROCESS({
        for (std::uint64_t i = 0; i < chunks_; ++i) {
            state_ = burnCpu(state_ + currentTick() + i, rounds_);
            if (i + 1 != chunks_) {
                co_await switchOut();
            }
        }
        checksum_ ^= state_ + (currentTick() << 7);
    })

    std::uint64_t checksum() const {
        return checksum_ ^ state_;
    }

private:
    std::uint64_t rounds_;
    std::uint64_t chunks_;
    std::uint64_t state_;
    std::uint64_t checksum_ = 0;
};

class LightWork final : public Component {
public:
    LightWork(std::size_t index, std::uint64_t rounds, TickProbe& probe)
        : index_(index),
          rounds_(rounds),
          probe_(probe),
          state_(0xbb67ae8584caa73bULL ^ index) {}

    MAKE_PROCESS({
        state_ = burnCpu(state_ + currentTick() + index_, rounds_);
        checksum_ ^= state_ + (currentTick() << 11) + index_;
        probe_.recordLight(index_);
    })

    std::uint64_t checksum() const {
        return checksum_ ^ state_;
    }

private:
    std::size_t index_;
    std::uint64_t rounds_;
    TickProbe& probe_;
    std::uint64_t state_;
    std::uint64_t checksum_ = 0;
};

struct BenchConfig {
    std::uint64_t ticks = 80;
    std::uint64_t warmup_ticks = 4;
    std::uint64_t light_components = 16;
    std::uint64_t light_rounds = 160000;
    std::uint64_t heavy_chunks = 8;
    std::uint64_t heavy_rounds_per_chunk = 240000;
    std::uint64_t load_balance_window = 4;
    std::uint64_t repeats = 3;
};

struct BenchResult {
    bool switching_out = false;
    bool load_balancing = false;
    bool heavy_first = true;
    std::size_t workers = 1;
    double seconds = 0.0;
    double ticks_per_second = 0.0;
    double mean_light_tail_ms = 0.0;
    std::uint64_t checksum = 0;
};

template <class HeavyT>
BenchResult runCase(const BenchConfig& config, std::size_t workers, bool switching_out,
                    bool load_balancing, bool heavy_first) {
    Simulator sim(load_balancing
                      ? SimulatorConfig{workers, static_cast<std::size_t>(
                                                     config.load_balance_window)}
                      : SimulatorConfig{workers, 0});
    TickProbe probe(static_cast<std::size_t>(config.light_components));

    HeavyT* heavy = nullptr;
    if (heavy_first) {
        heavy = &sim.createComponent<HeavyT>(
            config.heavy_rounds_per_chunk, config.heavy_chunks);
    }

    std::vector<LightWork*> lights;
    lights.reserve(static_cast<std::size_t>(config.light_components));
    for (std::uint64_t i = 0; i < config.light_components; ++i) {
        lights.push_back(&sim.createComponent<LightWork>(
            static_cast<std::size_t>(i), config.light_rounds, probe));
    }

    if (!heavy_first) {
        heavy = &sim.createComponent<HeavyT>(
            config.heavy_rounds_per_chunk, config.heavy_chunks);
    }

    for (std::uint64_t tick = 0; tick < config.warmup_ticks; ++tick) {
        probe.beginTick();
        sim.tick();
    }

    double light_tail_total_ms = 0.0;
    const auto start = Clock::now();
    for (std::uint64_t tick = 0; tick < config.ticks; ++tick) {
        probe.beginTick();
        sim.tick();
        light_tail_total_ms += probe.maxLightMs();
    }
    const auto end = Clock::now();

    std::uint64_t checksum = heavy->checksum();
    for (const auto* light : lights) {
        checksum ^= light->checksum();
    }

    const auto seconds = std::chrono::duration<double>(end - start).count();
    return BenchResult{
        switching_out,
        load_balancing,
        heavy_first,
        workers,
        seconds,
        static_cast<double>(config.ticks) / seconds,
        light_tail_total_ms / static_cast<double>(config.ticks),
        checksum,
    };
}

BenchResult bestOf(const BenchConfig& config, std::size_t workers, bool switching_out,
                   bool load_balancing, bool heavy_first) {
    BenchResult best{};
    best.switching_out = switching_out;
    best.load_balancing = load_balancing;
    best.heavy_first = heavy_first;
    best.workers = workers;

    for (std::uint64_t repeat = 0; repeat < config.repeats; ++repeat) {
        const auto result = switching_out
                                ? runCase<HeavySwitchingOut>(
                                      config, workers, switching_out, load_balancing,
                                      heavy_first)
                                : runCase<HeavyNoSwitchOut>(
                                      config, workers, switching_out, load_balancing,
                                      heavy_first);
        if (repeat == 0 || result.seconds < best.seconds) {
            best = result;
        }
    }

    return best;
}

void printResult(std::string_view name, const BenchResult& result,
                 const BenchResult& baseline) {
    std::cout << std::setw(14) << name
              << std::setw(8) << (result.load_balancing ? "on" : "off")
              << std::setw(9) << result.workers
              << std::setw(13) << std::fixed << std::setprecision(4)
              << result.seconds
              << std::setw(14) << std::fixed << std::setprecision(1)
              << result.ticks_per_second
              << std::setw(14) << std::fixed << std::setprecision(3)
              << (baseline.seconds / result.seconds)
              << std::setw(18) << std::fixed << std::setprecision(3)
              << result.mean_light_tail_ms
              << "    0x" << std::hex << result.checksum << std::dec << '\n';
}

void runScenario(const BenchConfig& config, bool heavy_first) {
    const auto baseline = bestOf(config, 1, false, false, heavy_first);
    const auto no_switch_out = bestOf(config, 2, false, false, heavy_first);
    const auto no_switch_out_lb = bestOf(config, 2, false, true, heavy_first);
    const auto switching_out = bestOf(config, 2, true, false, heavy_first);
    const auto switching_out_lb = bestOf(config, 2, true, true, heavy_first);

    std::cout << '\n'
              << (heavy_first ? "heavy-first registration" : "heavy-last registration")
              << '\n';
    std::cout << std::setw(14) << "mode"
              << std::setw(8) << "lb"
              << std::setw(9) << "workers"
              << std::setw(13) << "seconds"
              << std::setw(14) << "ticks/s"
              << std::setw(14) << "speedup"
              << std::setw(18) << "light_tail_ms"
              << "    checksum\n";
    printResult("no-switch-out", baseline, baseline);
    printResult("no-switch-out", no_switch_out, baseline);
    printResult("no-switch-out", no_switch_out_lb, baseline);
    printResult("switch-out", switching_out, baseline);
    printResult("switch-out", switching_out_lb, baseline);

    std::cout << "lb_over_plain_no_switch_out="
              << std::fixed << std::setprecision(3)
              << (no_switch_out.seconds / no_switch_out_lb.seconds) << "x\n";
    std::cout << "switch_out_lb_over_plain_no_switch_out="
              << std::fixed << std::setprecision(3)
              << (no_switch_out.seconds / switching_out_lb.seconds) << "x\n";
    std::cout << "switch_out_lb_light_tail_reduction="
              << std::fixed << std::setprecision(3)
              << (no_switch_out.mean_light_tail_ms / switching_out_lb.mean_light_tail_ms)
              << "x\n";
}

} // namespace

int main(int argc, char** argv) {
    BenchConfig config;
    config.ticks = parseArg(argv, argc, 1, config.ticks);
    config.light_components = parseArg(argv, argc, 2, config.light_components);
    config.light_rounds = parseArg(argv, argc, 3, config.light_rounds);
    config.heavy_rounds_per_chunk = parseArg(argv, argc, 4, config.heavy_rounds_per_chunk);
    config.heavy_chunks = parseArg(argv, argc, 5, config.heavy_chunks);
    config.repeats = parseArg(argv, argc, 6, config.repeats);

    std::cout << "2-worker cooperative switchOut fairness benchmark\n";
    std::cout << "one heavy component A is compared against a light-task backlog\n";
    std::cout << "A is heavier than any single light component, but the light backlog is "
                 "larger than A\n";
    std::cout << "switchOut is expected to improve light-task tail latency, not to parallelize A\n";
    std::cout << "ticks=" << config.ticks
              << ", warmup_ticks=" << config.warmup_ticks
              << ", light_components=" << config.light_components
              << ", light_rounds=" << config.light_rounds
              << ", heavy_chunks=" << config.heavy_chunks
              << ", heavy_rounds_per_chunk=" << config.heavy_rounds_per_chunk
              << ", load_balance_window=" << config.load_balance_window
              << ", repeats=" << config.repeats << '\n';
    std::cout << "usage: ./switch_out_balance_bench [ticks] [light_components] "
                 "[light_rounds] [heavy_rounds_per_chunk] [heavy_chunks] [repeats]\n";

    runScenario(config, true);
    runScenario(config, false);
}
