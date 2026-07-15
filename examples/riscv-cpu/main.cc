#include "core_state.hh"
#include "inst.hh"
#include "memory.hh"
#include "params.hh"
#include "program.hh"
#include "sim.hh"
#include "stages.hh"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::size_t parseTraceLimit(const std::string& arg) {
    const auto prefix = std::string{"--trace-limit="};
    if (arg.rfind(prefix, 0) != 0) {
        throw std::runtime_error("invalid trace limit option: " + arg);
    }
    return static_cast<std::size_t>(std::stoull(arg.substr(prefix.size())));
}

coropulse::TickId parseMaxTicks(const std::string& arg) {
    const auto prefix = std::string{"--max-ticks="};
    if (arg.rfind(prefix, 0) != 0) {
        throw std::runtime_error("invalid max ticks option: " + arg);
    }
    return static_cast<coropulse::TickId>(std::stoull(arg.substr(prefix.size())));
}

} // namespace

int main(int argc, char** argv) {
    const coropulse::Params params = {
        {"sim", {
            {"workers", 4},
            {"max_ticks", 0},
        }},
        {"core", {
            {"physical_registers", 64},
            {"issue_capacity", 16},
            {"commit_width", 1},
        }},
        {"memory", {
            {"data_bytes", 16 * 1024 * 1024},
            {"initial_word0", 10},
        }},
        {"program", {
            {"synthetic_instructions", 0},
            {"print_limit", 32},
        }},
    };

    std::string raw_path;
    bool trace = false;
    std::size_t trace_limit = 0;
    coropulse::TickId max_ticks_override = 0;
    for (int i = 1; i < argc; ++i) {
        const auto arg = std::string(argv[i]);
        if (arg == "--trace") {
            trace = true;
        } else if (arg.rfind("--trace-limit=", 0) == 0) {
            trace = true;
            trace_limit = parseTraceLimit(arg);
        } else if (arg.rfind("--max-ticks=", 0) == 0) {
            max_ticks_override = parseMaxTicks(arg);
        } else if (raw_path.empty()) {
            raw_path = arg;
        } else {
            throw std::runtime_error("unexpected argument: " + arg);
        }
    }
    const bool raw_program = !raw_path.empty();
    const auto synthetic_instructions =
        params["program"]["synthetic_instructions"].as<std::size_t>();
    const bool synthetic = !raw_program && synthetic_instructions != 0;

    std::vector<std::uint8_t> raw_image;
    std::vector<riscv_cpu::StaticInst> program;
    if (raw_program) {
        raw_image = riscv_cpu::loadRawImage(raw_path);
    } else {
        program = synthetic ? riscv_cpu::buildSyntheticProgram(synthetic_instructions)
                            : riscv_cpu::buildProgram();
    }
    const auto instruction_count =
        raw_program ? raw_image.size() / 4 : program.size();

    const auto configured_max_ticks =
        max_ticks_override != 0 ? max_ticks_override
                                : params["sim"]["max_ticks"].as<coropulse::TickId>();
    const auto tick_limit =
        configured_max_ticks != 0
            ? configured_max_ticks
            : (raw_program
                   ? static_cast<coropulse::TickId>(100000000)
                   : static_cast<coropulse::TickId>(instruction_count * 12 + 1000));

    auto make_sram = [&]() {
        if (raw_program) {
            return riscv_cpu::SimpleSram(
                raw_image, params["memory"]["data_bytes"].as<std::size_t>());
        }
        return riscv_cpu::SimpleSram(
            program, params["memory"]["data_bytes"].as<std::size_t>());
    };
    auto sram = make_sram();
    if (!raw_program) {
        sram.store64(0, params["memory"]["initial_word0"].as<std::uint64_t>());
    }

    riscv_cpu::CoreState core(params["core"]["physical_registers"].as<std::size_t>());
    riscv_cpu::DynInstPool inst_pool;
    coropulse::Simulator sim(params["sim"]["workers"].as<std::size_t>());

    auto& fetch = sim.createComponent<riscv_cpu::FetchStage>(sram, inst_pool);
    auto& decode = sim.createComponent<riscv_cpu::DecodeStage>();
    auto& rename = sim.createComponent<riscv_cpu::RenameStage>(core);
    auto& issue = sim.createComponent<riscv_cpu::IssueStage>(
        params["core"]["issue_capacity"].as<std::size_t>());
    auto& execute = sim.createComponent<riscv_cpu::ExecuteStage>(sram);
    auto& commit = sim.createComponent<riscv_cpu::CommitStage>(
        core, sram, params["core"]["commit_width"].as<std::size_t>(),
        trace ? &std::cerr : nullptr, trace_limit);

    sim.connect(fetch.out, decode.in);
    sim.connect(decode.out, rename.in);
    sim.connect(rename.out, issue.rename_in, commit.dispatch_in);
    sim.connect(issue.issue_out, execute.issue_in);
    sim.connect(execute.completion_out, issue.completion_in, commit.completion_in);
    sim.connect(commit.redirect_out, fetch.redirect_in, decode.redirect_in,
                rename.redirect_in, issue.redirect_in, execute.redirect_in);
    sim.connect(issue.can_accept, rename.issue_can_accept);
    sim.connect(rename.can_accept, decode.rename_can_accept);
    sim.connect(decode.can_accept, fetch.decode_can_accept);

    coropulse::TickId ticks = 0;
    bool completed = false;
    const auto start = std::chrono::steady_clock::now();
    for (; ticks < tick_limit; ++ticks) {
        sim.tick();
        completed = fetch.halted() && core.inFlightCount() == 0 &&
                    fetch.redirectCount() == commit.redirectCount();
        if (completed) {
            break;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;

    std::cout << "simple cpu microarchitecture example\n";
    std::cout << "source=" << (raw_program ? raw_path : (synthetic ? "synthetic" : "builtin"))
              << ", workers=" << params["sim"]["workers"].as<std::size_t>()
              << ", instructions=" << instruction_count
              << ", physical_registers=" << core.physicalRegisterCount()
              << ", issue_capacity=" << params["core"]["issue_capacity"].as<std::size_t>()
              << ", ticks=" << (ticks + 1)
              << ", fetched=" << fetch.fetchedCount()
              << ", committed=" << core.committedCount()
              << ", ipc=" << static_cast<double>(core.committedCount()) /
                               static_cast<double>(ticks + 1)
              << ", elapsed_ms=" << (elapsed.count() * 1000.0)
              << ", committed_per_sec="
              << (static_cast<double>(core.committedCount()) / elapsed.count())
              << ", worker_idle=" << (sim.workerIdleRatio() * 100.0) << "%\n";
    std::cout << "fetch=" << fetch.fetchedCount()
              << ", decode=" << decode.decodedCount()
              << ", rename=" << rename.renamedCount()
              << ", issue_accept=" << issue.acceptedCount()
              << ", issue=" << issue.issuedCount()
              << ", execute_accept=" << execute.acceptedCount()
              << ", execute_complete=" << execute.completedCount()
              << ", commit=" << commit.retiredCount()
              << ", commit_redirects=" << commit.redirectCount()
              << ", issue_output_stalls=" << issue.outputStalls() << '\n';
    std::cout << "fetch_backpressure_stalls=" << fetch.backpressureStalls()
              << ", fetch_control_stalls=" << fetch.controlStalls()
              << ", redirects=" << fetch.redirectCount()
              << ", decode_backpressure_stalls=" << decode.backpressureStalls()
              << ", rename_resource_stalls=" << rename.resourceStalls()
              << ", rename_issue_backpressure_stalls="
              << rename.issueBackpressureStalls()
              << ", free_physical_registers=" << core.freePhysicalRegisters() << '\n';
    std::cout << "x1=" << core.registerValue(1)
              << ", x5=" << core.registerValue(5)
              << ", x6=" << core.registerValue(6)
              << ", x7=" << core.registerValue(7)
              << ", x10=" << core.registerValue(10)
              << ", mem[8]=" << sram.load64(8) << '\n';
    if (raw_program) {
        std::cout << "program: raw image, lazy decoded by fetch\n";
    } else {
        riscv_cpu::printProgram(program, std::cout,
                                params["program"]["print_limit"].as<std::size_t>());
    }

    if (!completed) {
        std::cerr << "cpu example did not drain before max_ticks="
                  << tick_limit << '\n';
        return 1;
    }

    bool valid_state = true;
    if (synthetic) {
        for (int reg = 1; reg < 32; ++reg) {
            if (core.registerValue(reg) !=
                riscv_cpu::syntheticRegisterValue(instruction_count, reg)) {
                valid_state = false;
                break;
            }
        }
    } else if (!raw_program) {
        valid_state = core.registerValue(7) == 30 && sram.load64(8) == 30;
    }

    if (!valid_state) {
        std::cerr << "cpu example produced an unexpected architectural state\n";
        return 1;
    }

    return 0;
}
