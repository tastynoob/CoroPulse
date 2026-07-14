#include "cpas/sim.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

using namespace cpas;

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

class WorkComponent final : public Component {
public:
    WorkComponent(std::uint64_t id, std::uint64_t work_rounds)
        : id_(id),
          work_rounds_(work_rounds),
          state_(0x123456789abcdef0ULL ^ (id * 0x100000001b3ULL)) {}

    Task<void> tick(TickContext& ctx) override {
        state_ = burnCpu(state_ + ctx.tick() + id_, work_rounds_);
        checksum_ ^= state_ + (ctx.tick() << 11) + id_;
        co_return;
    }

    std::uint64_t checksum() const {
        return checksum_ ^ state_;
    }

private:
    std::uint64_t id_;
    std::uint64_t work_rounds_;
    std::uint64_t state_;
    std::uint64_t checksum_ = 0;
};

struct BenchConfig {
    std::uint64_t ticks = 200;
    std::uint64_t warmup_ticks = 2;
    std::uint64_t light_components = 48;
    std::uint64_t light_rounds = 40000;
    std::uint64_t heavy_rounds = 600000;
    std::uint64_t repeats = 3;
    std::uint64_t window_ticks = 8;
};

struct BenchResult {
    bool load_balancing = false;
    double seconds = 0.0;
    double ticks_per_second = 0.0;
    std::uint64_t checksum = 0;
};

BenchResult runOnce(const BenchConfig& config, bool load_balancing) {
    Runtime runtime(4);
    if (load_balancing) {
        runtime.enableLoadBalancing(static_cast<std::size_t>(config.window_ticks));
    }

    std::vector<std::unique_ptr<WorkComponent>> components;
    components.reserve(static_cast<std::size_t>(config.light_components + 1));

    for (std::uint64_t i = 0; i < config.light_components; ++i) {
        components.push_back(std::make_unique<WorkComponent>(i, config.light_rounds));
        runtime.addComponent(*components.back());
    }

    components.push_back(std::make_unique<WorkComponent>(
        config.light_components, config.heavy_rounds));
    runtime.addComponent(*components.back());

    for (std::uint64_t i = 0; i < config.warmup_ticks; ++i) {
        runtime.runTick();
    }

    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < config.ticks; ++i) {
        runtime.runTick();
    }
    const auto end = std::chrono::steady_clock::now();

    std::uint64_t checksum = 0;
    for (const auto& component : components) {
        checksum ^= component->checksum();
    }

    const auto seconds = std::chrono::duration<double>(end - start).count();
    return BenchResult{
        load_balancing,
        seconds,
        static_cast<double>(config.ticks) / seconds,
        checksum,
    };
}

BenchResult bestOf(const BenchConfig& config, bool load_balancing) {
    BenchResult best{};
    best.load_balancing = load_balancing;

    for (std::uint64_t repeat = 0; repeat < config.repeats; ++repeat) {
        const auto result = runOnce(config, load_balancing);
        if (repeat == 0 || result.seconds < best.seconds) {
            best = result;
        }
    }

    return best;
}

void printResult(std::string_view name, const BenchResult& result) {
    std::cout << std::setw(16) << name
              << std::setw(14) << std::fixed << std::setprecision(4)
              << result.seconds
              << std::setw(16) << std::fixed << std::setprecision(1)
              << result.ticks_per_second
              << "    0x" << std::hex << result.checksum << std::dec << '\n';
}

} // namespace

int main(int argc, char** argv) {
    BenchConfig config;
    config.ticks = parseArg(argv, argc, 1, config.ticks);
    config.light_components = parseArg(argv, argc, 2, config.light_components);
    config.light_rounds = parseArg(argv, argc, 3, config.light_rounds);
    config.heavy_rounds = parseArg(argv, argc, 4, config.heavy_rounds);
    config.repeats = parseArg(argv, argc, 5, config.repeats);

    const auto without = bestOf(config, false);
    const auto with = bestOf(config, true);
    const auto speedup = without.seconds / with.seconds;

    std::cout << "4-worker load balancing benchmark\n";
    std::cout << "one heavy component is registered after all light components\n";
    std::cout << "ticks=" << config.ticks
              << ", warmup_ticks=" << config.warmup_ticks
              << ", light_components=" << config.light_components
              << ", light_rounds=" << config.light_rounds
              << ", heavy_rounds=" << config.heavy_rounds
              << ", repeats=" << config.repeats
              << ", window_ticks=" << config.window_ticks << '\n';
    std::cout << "usage: ./load_balance_bench [ticks] [light_components] "
                 "[light_rounds] [heavy_rounds] [repeats]\n";
    std::cout << std::setw(16) << "mode"
              << std::setw(14) << "seconds"
              << std::setw(16) << "ticks/s"
              << "    checksum\n";
    printResult("off", without);
    printResult("on", with);
    std::cout << "speedup_off_over_on=" << std::fixed << std::setprecision(3)
              << speedup << "x\n";
}
