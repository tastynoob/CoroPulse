#include "backend_pipeline.hh"

#include "exec_context.hh"
#include "isa_execute.hh"

#include <utility>

namespace riscv_cpu {

ExecutePipe::ExecutePipe(CoreState& core, SimpleSram& sram)
    : core_(core), sram_(sram) {}

void ExecutePipe::accept(InstBundle&& bundle, coropulse::TickId tick,
                         BackendStats& stats) {
    for (auto* inst : bundle) {
        inst->executeState().done_tick = tick + inst->staticInst().latency;
        executing_.push_back(Executing{
            inst,
            inst->executeState().done_tick,
        });
    }
    stats.execute_accepted += bundle.size();
}

InstBundle ExecutePipe::collectCompletions(coropulse::TickId tick,
                                           BackendStats& stats) {
    InstBundle completed;
    if (!pending_completion_.empty()) {
        completed = std::move(pending_completion_);
        pending_completion_.clear();
        stats.execute_completed += completed.size();
    }

    completeReadyUops(tick);
    return completed;
}

void ExecutePipe::clear() {
    executing_.clear();
    pending_completion_.clear();
}

void ExecutePipe::completeReadyUops(coropulse::TickId tick) {
    if (!pending_completion_.empty()) {
        return;
    }

    for (auto iter = executing_.begin(); iter != executing_.end();) {
        if (iter->done_tick <= tick) {
            execute(iter->inst, tick);
            pending_completion_.push_back(iter->inst);
            executing_.erase(iter);
        } else {
            ++iter;
        }
    }
}

void ExecutePipe::execute(DynInstPtr inst, coropulse::TickId tick) {
    ExecContext context(core_, sram_, *inst, tick);
    executeInst(context);
}

} // namespace riscv_cpu
