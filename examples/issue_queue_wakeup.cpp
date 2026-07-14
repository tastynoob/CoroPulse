#include "sim.hh"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

using namespace coropulse;

namespace {

struct Instruction {
    int id = 0;
    int latency = 1;
    std::vector<int> deps;
};

struct IssuedInstruction {
    int id = 0;
    int latency = 1;
};

struct WakeupData {
    std::vector<int> completed_ids;
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

std::vector<Instruction> buildProgram(int instruction_count, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> latency_dist(1, 5);
    std::uniform_int_distribution<int> percent_dist(0, 99);

    std::vector<Instruction> program;
    program.reserve(instruction_count);

    for (int id = 0; id < instruction_count; ++id) {
        Instruction inst;
        inst.id = id;
        inst.latency = latency_dist(rng);

        if (id > 0) {
            std::unordered_set<int> deps;

            // Keep a visible dependency-chain backbone, then add a random older dependency.
            if (percent_dist(rng) < 75) {
                deps.insert(id - 1);
            } else {
                std::uniform_int_distribution<int> dep_dist(0, id - 1);
                deps.insert(dep_dist(rng));
            }

            if (id > 2 && percent_dist(rng) < 35) {
                std::uniform_int_distribution<int> dep_dist(0, id - 1);
                deps.insert(dep_dist(rng));
            }

            inst.deps.assign(deps.begin(), deps.end());
            std::sort(inst.deps.begin(), inst.deps.end());
        }

        program.push_back(std::move(inst));
    }

    return program;
}

std::vector<std::vector<int>> buildConsumers(const std::vector<Instruction>& program) {
    std::vector<std::vector<int>> consumers(program.size());
    for (const auto& inst : program) {
        for (const auto dep : inst.deps) {
            consumers[dep].push_back(inst.id);
        }
    }
    return consumers;
}

class IssueQueue final : public Component {
public:
    SignalInput<bool> need_wakeup{"need_wakeup"};
    SignalInput<WakeupData> wakeup_data{"wakeup_data"};
    Output<IssuedInstruction> issue_out{"issue_out"};

    IssueQueue(const std::vector<Instruction>& program,
               const std::vector<std::vector<int>>& consumers)
        : program_(program),
          consumers_(consumers),
          remaining_deps_(program.size(), 0),
          issued_(program.size(), false),
          completed_(program.size(), false) {
        for (const auto& inst : program_) {
            remaining_deps_[inst.id] = static_cast<int>(inst.deps.size());
        }
    }

    Task<void> tick() override {
        const bool wakeup_needed = co_await need_wakeup.read();

        std::vector<bool> became_ready_this_tick(program_.size(), false);
        if (wakeup_needed) {
            const auto data = co_await wakeup_data.read();
            ++wakeup_ticks_;

            for (const auto completed_id : data.completed_ids) {
                if (completed_[completed_id]) {
                    continue;
                }
                completed_[completed_id] = true;
                ++completed_seen_;

                for (const auto consumer_id : consumers_[completed_id]) {
                    assert(remaining_deps_[consumer_id] > 0);
                    --remaining_deps_[consumer_id];
                    if (remaining_deps_[consumer_id] == 0) {
                        became_ready_this_tick[consumer_id] = true;
                    }
                }
            }
        }

        const auto candidate = pickReadyInstruction();
        if (!candidate) {
            co_return;
        }

        const auto& inst = program_[*candidate];
        if (!issue_out.write(IssuedInstruction{inst.id, inst.latency})) {
            ++channel_full_stalls_;
            co_return;
        }

        issued_[inst.id] = true;
        ++issued_count_;

        if (became_ready_this_tick[inst.id]) {
            ++same_tick_wakeup_issues_;
        }

        co_return;
    }

    bool allIssued() const {
        return issued_count_ == program_.size();
    }

    bool allCompletedSeen() const {
        return completed_seen_ == program_.size();
    }

    std::size_t issuedCount() const { return issued_count_; }
    std::size_t completedSeen() const { return completed_seen_; }
    std::size_t wakeupTicks() const { return wakeup_ticks_; }
    std::size_t sameTickWakeupIssues() const { return same_tick_wakeup_issues_; }
    std::size_t channelFullStalls() const { return channel_full_stalls_; }

private:
    std::optional<int> pickReadyInstruction() const {
        for (const auto& inst : program_) {
            if (!issued_[inst.id] && remaining_deps_[inst.id] == 0) {
                return inst.id;
            }
        }
        return std::nullopt;
    }

    const std::vector<Instruction>& program_;
    const std::vector<std::vector<int>>& consumers_;
    std::vector<int> remaining_deps_;
    std::vector<bool> issued_;
    std::vector<bool> completed_;
    std::size_t issued_count_ = 0;
    std::size_t completed_seen_ = 0;
    std::size_t wakeup_ticks_ = 0;
    std::size_t same_tick_wakeup_issues_ = 0;
    std::size_t channel_full_stalls_ = 0;
};

class ExecuteUnit final : public Component {
public:
    Input<IssuedInstruction> issue_in{"issue_in"};
    SignalOutput<bool> need_wakeup{"need_wakeup"};
    SignalOutput<WakeupData> wakeup_data{"wakeup_data"};

    explicit ExecuteUnit(std::uint64_t execute_work_rounds)
        : execute_work_rounds_(execute_work_rounds) {}

    Task<void> tick() override {
        WakeupData wakeup;

        const auto active_count = static_cast<std::uint64_t>(executing_.size());
        const auto work_rounds = execute_work_rounds_ * (active_count + 1);
        execute_checksum_ = burnCpu(execute_checksum_ + currentTick() + active_count,
                                    work_rounds);

        auto out = executing_.begin();
        while (out != executing_.end()) {
            if (out->done_tick <= currentTick()) {
                wakeup.completed_ids.push_back(out->id);
                out = executing_.erase(out);
            } else {
                ++out;
            }
        }

        if (!wakeup.completed_ids.empty()) {
            wakeup_data.set(wakeup);
            need_wakeup.set(true);
            completed_count_ += wakeup.completed_ids.size();
            ++wakeup_broadcasts_;
        } else {
            need_wakeup.set(false);
        }

        auto issued = issue_in.read();
        if (issued) {
            executing_.push_back(Executing{
                issued->id,
                currentTick() + static_cast<TickId>(issued->latency),
            });
            ++accepted_count_;
        }

        co_return;
    }

    bool idle() const { return executing_.empty(); }
    std::size_t acceptedCount() const { return accepted_count_; }
    std::size_t completedCount() const { return completed_count_; }
    std::size_t wakeupBroadcasts() const { return wakeup_broadcasts_; }
    std::uint64_t checksum() const { return execute_checksum_; }

private:
    struct Executing {
        int id = 0;
        TickId done_tick = 0;
    };

    std::uint64_t execute_work_rounds_;
    std::vector<Executing> executing_;
    std::uint64_t execute_checksum_ = 0x123456789abcdef0ULL;
    std::size_t accepted_count_ = 0;
    std::size_t completed_count_ = 0;
    std::size_t wakeup_broadcasts_ = 0;
};

} // namespace

int main(int argc, char** argv) {
    const auto instruction_count =
        static_cast<int>(parseArg(argv, argc, 1, 40));
    const auto seed = parseArg(argv, argc, 2, 0xC0F0'2026ULL);
    const auto workers = parseArg(argv, argc, 3, 4);
    const auto max_ticks = parseArg(argv, argc, 4, 2000);
    const auto execute_work_rounds = parseArg(argv, argc, 5, 2000);

    const auto program = buildProgram(instruction_count, seed);
    const auto consumers = buildConsumers(program);

    Simulator sim(static_cast<std::size_t>(workers));
    auto& issue_queue = sim.createComponent<IssueQueue>(program, consumers);
    auto& execute_unit = sim.createComponent<ExecuteUnit>(execute_work_rounds);
    sim.connect(execute_unit.need_wakeup, issue_queue.need_wakeup);
    sim.connect(execute_unit.wakeup_data, issue_queue.wakeup_data);
    sim.connect(issue_queue.issue_out, execute_unit.issue_in);

    TickId ticks = 0;
    for (; ticks < max_ticks; ++ticks) {
        sim.tick();
        if (issue_queue.allIssued() && issue_queue.allCompletedSeen() && execute_unit.idle()) {
            break;
        }
    }

    std::cout << "issue queue same-tick wakeup example\n";
    std::cout << "instructions=" << instruction_count << ", seed=" << seed
              << ", workers=" << workers
              << ", execute_work_rounds=" << execute_work_rounds << '\n';
    std::cout << "ticks=" << (ticks + 1)
              << ", issued=" << issue_queue.issuedCount()
              << ", execute_accepted=" << execute_unit.acceptedCount()
              << ", completed=" << execute_unit.completedCount() << '\n';
    std::cout << "wakeup_ticks=" << issue_queue.wakeupTicks()
              << ", wakeup_broadcasts=" << execute_unit.wakeupBroadcasts()
              << ", same_tick_wakeup_issues="
              << issue_queue.sameTickWakeupIssues()
              << ", channel_full_stalls=" << issue_queue.channelFullStalls()
              << ", execute_checksum=0x" << std::hex << execute_unit.checksum()
              << std::dec
              << '\n';

    std::cout << "first instructions:\n";
    const auto preview = std::min<std::size_t>(program.size(), 12);
    for (std::size_t i = 0; i < preview; ++i) {
        const auto& inst = program[i];
        std::cout << "  i" << inst.id << " latency=" << inst.latency << " deps=[";
        for (std::size_t dep = 0; dep < inst.deps.size(); ++dep) {
            if (dep != 0) {
                std::cout << ',';
            }
            std::cout << inst.deps[dep];
        }
        std::cout << "]\n";
    }

    if (!issue_queue.allIssued() || !issue_queue.allCompletedSeen() || !execute_unit.idle()) {
        std::cerr << "simulation did not finish before max_ticks=" << max_ticks << '\n';
        return 1;
    }

    if (issue_queue.sameTickWakeupIssues() == 0) {
        std::cerr << "same-tick wakeup did not issue any newly-ready instruction\n";
        return 1;
    }

    return 0;
}
