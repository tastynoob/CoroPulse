#include "core_state.hh"
#include "inst.hh"
#include "memory.hh"
#include "params.hh"
#include "program.hh"
#include "sim.hh"
#include "stages.hh"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

std::size_t parseTraceLimit(const std::string& arg) {
    const auto prefix = std::string{"--trace-limit="};
    if (arg.rfind(prefix, 0) != 0) {
        throw std::runtime_error("invalid trace limit option: " + arg);
    }
    return static_cast<std::size_t>(std::stoull(arg.substr(prefix.size())));
}

std::size_t parseSizeOption(const std::string& arg, const std::string& prefix) {
    if (arg.rfind(prefix, 0) != 0) {
        throw std::runtime_error("invalid size option: " + arg);
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
            {"fetch_width", 4},
            {"fetch_decode_fifo_capacity", 16},
            {"decode_width", 4},
            {"rename_width", 4},
            {"issue_width", 4},
            {"commit_width", 4},
        }},
        {"memory", {
            {"data_bytes", 16 * 1024 * 1024},
        }},
    };

    std::string raw_path;
    bool trace = false;
    bool load_balance = true;
    std::size_t trace_limit = 0;
    std::size_t workers_override = 0;
    coropulse::TickId max_ticks_override = 0;
    std::size_t fetch_width_override = 0;
    std::size_t fifo_capacity_override = 0;
    std::size_t decode_width_override = 0;
    std::size_t rename_width_override = 0;
    std::size_t issue_capacity_override = 0;
    std::size_t issue_width_override = 0;
    std::size_t commit_width_override = 0;
    for (int i = 1; i < argc; ++i) {
        const auto arg = std::string(argv[i]);
        if (arg == "--trace") {
            trace = true;
        } else if (arg == "--no-load-balance") {
            load_balance = false;
        } else if (arg.rfind("--trace-limit=", 0) == 0) {
            trace = true;
            trace_limit = parseTraceLimit(arg);
        } else if (arg.rfind("--max-ticks=", 0) == 0) {
            max_ticks_override = parseMaxTicks(arg);
        } else if (arg.rfind("--workers=", 0) == 0) {
            workers_override = parseSizeOption(arg, "--workers=");
        } else if (arg.rfind("--fetch-width=", 0) == 0) {
            fetch_width_override = parseSizeOption(arg, "--fetch-width=");
        } else if (arg.rfind("--fifo-capacity=", 0) == 0) {
            fifo_capacity_override = parseSizeOption(arg, "--fifo-capacity=");
        } else if (arg.rfind("--decode-width=", 0) == 0) {
            decode_width_override = parseSizeOption(arg, "--decode-width=");
        } else if (arg.rfind("--rename-width=", 0) == 0) {
            rename_width_override = parseSizeOption(arg, "--rename-width=");
        } else if (arg.rfind("--issue-capacity=", 0) == 0) {
            issue_capacity_override = parseSizeOption(arg, "--issue-capacity=");
        } else if (arg.rfind("--issue-width=", 0) == 0) {
            issue_width_override = parseSizeOption(arg, "--issue-width=");
        } else if (arg.rfind("--commit-width=", 0) == 0) {
            commit_width_override = parseSizeOption(arg, "--commit-width=");
        } else if (raw_path.empty()) {
            raw_path = arg;
        } else {
            throw std::runtime_error("unexpected argument: " + arg);
        }
    }
    if (raw_path.empty()) {
        throw std::runtime_error(
            "missing raw program image path; usage: riscv_cpu <program.bin> [options]");
    }

    auto raw_image = riscv_cpu::loadRawImage(raw_path);
    const auto instruction_count = raw_image.size() / 4;

    const auto configured_max_ticks =
        max_ticks_override != 0 ? max_ticks_override
                                : params["sim"]["max_ticks"].as<coropulse::TickId>();
    const auto tick_limit =
        configured_max_ticks != 0
            ? configured_max_ticks
            : static_cast<coropulse::TickId>(100000000);

    riscv_cpu::SimpleSram sram(
        std::move(raw_image), params["memory"]["data_bytes"].as<std::size_t>());

    riscv_cpu::CoreState core(params["core"]["physical_registers"].as<std::size_t>());
    riscv_cpu::DynInstPool inst_pool;
    const auto workers = workers_override != 0
                             ? workers_override
                             : params["sim"]["workers"].as<std::size_t>();
    coropulse::Simulator sim(workers);

    const auto issue_capacity =
        issue_capacity_override != 0
            ? issue_capacity_override
            : params["core"]["issue_capacity"].as<std::size_t>();
    const auto fetch_width =
        fetch_width_override != 0 ? fetch_width_override
                                  : params["core"]["fetch_width"].as<std::size_t>();
    const auto fifo_capacity =
        fifo_capacity_override != 0
            ? fifo_capacity_override
            : params["core"]["fetch_decode_fifo_capacity"].as<std::size_t>();
    const auto decode_width =
        decode_width_override != 0 ? decode_width_override
                                   : params["core"]["decode_width"].as<std::size_t>();
    const auto rename_width =
        rename_width_override != 0 ? rename_width_override
                                   : params["core"]["rename_width"].as<std::size_t>();
    const auto issue_width =
        issue_width_override != 0 ? issue_width_override
                                  : params["core"]["issue_width"].as<std::size_t>();
    const auto commit_width =
        commit_width_override != 0 ? commit_width_override
                                   : params["core"]["commit_width"].as<std::size_t>();

    auto& fetch = sim.createComponent<riscv_cpu::FetchStage>(
        sram, inst_pool, fetch_width);
    auto& backend = sim.createComponent<riscv_cpu::BackendStage>(
        core, sram, rename_width, fifo_capacity, fetch_width, decode_width,
        issue_capacity, rename_width, issue_width, commit_width,
        trace ? &std::cerr : nullptr, trace_limit);

    sim.connect(fetch.out, backend.in);
    sim.connect(backend.predictor_update_out, fetch.predictor_update_in);
    sim.connect(backend.redirect_out, fetch.redirect_in);
    sim.connect(backend.can_accept, fetch.backend_can_accept);

    if (load_balance) {
        sim.enableLoadBalancing(5);
    }

    coropulse::TickId ticks = 0;
    bool completed = false;
    const auto start = std::chrono::steady_clock::now();
    for (; ticks < tick_limit; ++ticks) {
        sim.tick();
        completed = fetch.architecturalHalted() && core.inFlightCount() == 0 &&
                    fetch.stats.redirects == backend.stats.redirects;
        if (completed) {
            break;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    const auto profiling = sim.profilingStats();
    const auto scheduler_elapsed_ms =
        std::chrono::duration<double, std::milli>(profiling.elapsed_time).count();
    const auto worker_capacity_ms =
        std::chrono::duration<double, std::milli>(profiling.worker_capacity_time).count();
    const auto worker_idle_ms =
        std::chrono::duration<double, std::milli>(profiling.worker_idle_time).count();
    const auto task_active_ms =
        std::chrono::duration<double, std::milli>(profiling.task_active_time).count();
    const auto accounted = profiling.worker_idle_time + profiling.task_active_time;
    const auto unattributed =
        profiling.worker_capacity_time > accounted
            ? profiling.worker_capacity_time - accounted
            : coropulse::Scheduler::Duration{0};
    const auto unattributed_ms =
        std::chrono::duration<double, std::milli>(unattributed).count();

    std::cout << "simple cpu microarchitecture example\n";
    std::cout << "source=" << raw_path
              << ", workers=" << workers
              << ", instructions=" << instruction_count
              << ", physical_registers=" << core.physicalRegisterCount()
              << ", issue_capacity=" << issue_capacity
              << ", fetch_width=" << fetch_width
              << ", fetch_decode_fifo_capacity=" << fifo_capacity
              << ", decode_width=" << decode_width
              << ", rename_width=" << rename_width
              << ", issue_width=" << issue_width
              << ", commit_width=" << commit_width
              << ", load_balance=" << (load_balance ? "on" : "off")
              << ", ticks=" << (ticks + 1)
              << ", fetched=" << fetch.stats.fetched
              << ", committed=" << core.committedCount()
              << ", ipc=" << static_cast<double>(core.committedCount()) /
                               static_cast<double>(ticks + 1)
              << ", elapsed_ms=" << (elapsed.count() * 1000.0)
              << ", committed_per_sec="
              << (static_cast<double>(core.committedCount()) / elapsed.count())
              << ", worker_idle=" << (sim.workerIdleRatio() * 100.0) << "%\n";
    std::cout << "fetch=" << fetch.stats.fetched
              << ", decode=" << backend.stats.decoded
              << ", rename=" << backend.stats.renamed
              << ", issue_accept=" << backend.stats.accepted
              << ", issue=" << backend.stats.issued
              << ", execute_accept=" << backend.stats.execute_accepted
              << ", execute_complete=" << backend.stats.execute_completed
              << ", commit=" << backend.stats.retired
              << ", commit_redirects=" << backend.stats.redirects
              << ", fetch_decode_fifo_max="
              << backend.stats.frontend_queue_max_occupancy
              << ", fetch_decode_fifo_stalls="
              << backend.stats.frontend_queue_stalls << '\n';
    std::cout << "fetch_backpressure_stalls="
              << fetch.stats.backpressure_stalls
              << ", fetch_control_stalls=" << fetch.stats.control_stalls
              << ", redirects=" << fetch.stats.redirects
              << ", decode_backpressure_stalls="
              << backend.stats.decode_backpressure_stalls
              << ", rename_resource_stalls=" << backend.stats.resource_stalls
              << ", rename_issue_backpressure_stalls="
              << backend.stats.issue_backpressure_stalls
              << ", free_physical_registers=" << core.freePhysicalRegisters() << '\n';
    const auto rate = [](std::size_t numerator, std::size_t denominator) {
        if (denominator == 0) {
            return 0.0;
        }
        return static_cast<double>(numerator) / static_cast<double>(denominator);
    };
    const auto& predictor = fetch.stats.predictor;
    const auto bp_mispred_rate =
        rate(predictor.mispredictions, predictor.updates);
    const auto bp_direction_mispred_rate =
        rate(predictor.direction_mispredictions, predictor.conditional_updates);
    const auto bp_target_mispred_rate =
        rate(predictor.target_mispredictions, predictor.updates);
    const auto micro_btb_hit_rate =
        rate(predictor.micro_btb_hits, predictor.micro_btb_lookups);
    const auto tage_provider_hit_rate =
        rate(predictor.tage_provider_hits, predictor.tage_lookups);
    const auto tage_override_rate =
        rate(predictor.tage_overrides, predictor.tage_lookups);
    std::cout << "bp_predictions=" << predictor.predictions
              << ", bp_conditional_predictions="
              << predictor.conditional_predictions
              << ", bp_updates=" << predictor.updates
              << ", bp_conditional_updates=" << predictor.conditional_updates
              << ", bp_taken_updates=" << predictor.taken_updates
              << ", bp_mispredictions=" << predictor.mispredictions
              << ", bp_mispred_rate=" << (bp_mispred_rate * 100.0) << "%"
              << ", bp_direction_mispredictions="
              << predictor.direction_mispredictions
              << ", bp_direction_mispred_rate="
              << (bp_direction_mispred_rate * 100.0) << "%"
              << ", bp_target_mispredictions="
              << predictor.target_mispredictions
              << ", bp_target_mispred_rate="
              << (bp_target_mispred_rate * 100.0) << "%\n";
    std::cout << "micro_btb_lookups=" << predictor.micro_btb_lookups
              << ", micro_btb_hits=" << predictor.micro_btb_hits
              << ", micro_btb_hit_rate="
              << (micro_btb_hit_rate * 100.0) << "%"
              << ", micro_btb_updates=" << predictor.micro_btb_updates
              << ", micro_btb_allocations="
              << predictor.micro_btb_allocations
              << ", tage_lookups=" << predictor.tage_lookups
              << ", tage_provider_hits=" << predictor.tage_provider_hits
              << ", tage_provider_hit_rate="
              << (tage_provider_hit_rate * 100.0) << "%"
              << ", tage_base_uses=" << predictor.tage_base_uses
              << ", tage_alternate_uses=" << predictor.tage_alternate_uses
              << ", tage_overrides=" << predictor.tage_overrides
              << ", tage_override_rate=" << (tage_override_rate * 100.0) << "%"
              << ", tage_allocations=" << predictor.tage_allocations
              << ", tage_useful_ages=" << predictor.tage_useful_ages
              << ", frontend_squashes=" << fetch.stats.frontend_squashes << '\n';
    std::cout << "x1=" << core.registerValue(1)
              << ", x5=" << core.registerValue(5)
              << ", x6=" << core.registerValue(6)
              << ", x7=" << core.registerValue(7)
              << ", x10=" << core.registerValue(10)
              << ", mem[8]=" << sram.load64(8) << '\n';
    std::cout << "scheduler_elapsed_ms=" << scheduler_elapsed_ms
              << ", worker_capacity_ms=" << worker_capacity_ms
              << ", task_active_ms=" << task_active_ms
              << ", task_active=" << (profiling.taskActiveRatio() * 100.0) << "%"
              << ", worker_idle_ms=" << worker_idle_ms
              << ", worker_idle=" << (profiling.idleRatio() * 100.0) << "%"
              << ", scheduler_unattributed_ms=" << unattributed_ms
              << ", scheduler_unattributed="
              << (profiling.unattributedRatio() * 100.0) << "%\n";
    std::cout << "scheduler_resumes=" << profiling.resume_count
              << ", scheduler_tick_done=" << profiling.tick_done_count
              << ", scheduler_deferred_resumes=" << profiling.deferred_resume_count
              << ", scheduler_sleep=" << profiling.sleep_count << '\n';
    std::cout << "program: raw image, lazy decoded by fetch\n";

    if (!completed) {
        std::cerr << "cpu example did not drain before max_ticks="
                  << tick_limit << '\n';
        return 1;
    }

    return 0;
}
