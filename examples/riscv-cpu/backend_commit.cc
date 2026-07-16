#include "backend_pipeline.hh"

#include <iomanip>
#include <ostream>
#include <utility>

namespace riscv_cpu {

CommitPipe::CommitPipe(CoreState& core, SimpleSram& sram,
                       std::size_t commit_width, std::ostream* trace_out,
                       std::size_t trace_limit)
    : core_(core),
      sram_(sram),
      commit_width_(commit_width),
      trace_out_(trace_out),
      trace_limit_(trace_limit) {}

bool CommitPipe::beginTick() {
    const bool flushing = flush_next_tick_;
    flush_next_tick_ = false;
    return flushing;
}

bool CommitPipe::hasPendingRedirect() const noexcept {
    return pending_redirect_.has_value();
}

bool CommitPipe::publishPendingRedirect(coropulse::Output<ControlRedirect>& output,
                                        BackendStats& stats) {
    if (!pending_redirect_) {
        return true;
    }

    if (output.write(*pending_redirect_)) {
        pending_redirect_.reset();
        flush_next_tick_ = true;
        ++stats.redirects;
        return true;
    }

    flush_next_tick_ = true;
    return false;
}

bool CommitPipe::publishPredictorUpdates(
    coropulse::Output<BranchUpdateBundle>& output) {
    if (pending_updates_.empty()) {
        return true;
    }

    if (!output.write(pending_updates_)) {
        return false;
    }

    pending_updates_.clear();
    return true;
}

void CommitPipe::queuePredictorUpdates(BranchUpdateBundle&& updates) {
    if (updates.empty()) {
        return;
    }

    if (pending_updates_.empty()) {
        pending_updates_ = std::move(updates);
        return;
    }

    pending_updates_.insert(pending_updates_.end(), updates.begin(), updates.end());
}

RetireResult CommitPipe::retire(coropulse::TickId tick, BackendStats& stats) {
    auto result = core_.retire(commit_width_);
    for (const auto& inst : result.trace) {
        traceRetired(tick, inst);
    }
    for (const auto& store : result.stores) {
        sram_.store(store.address, store.value, store.bytes);
    }
    stats.retired += result.retired;
    if (result.redirect) {
        pending_redirect_ = result.redirect;
    }
    return result;
}

void CommitPipe::markCompleted(const InstBundle& bundle) {
    for (const auto& completion : bundle) {
        core_.markCompleted(completion);
    }
}

void CommitPipe::dispatch(const InstBundle& bundle) {
    for (auto* inst : bundle) {
        core_.dispatchRenamed(inst);
    }
}

void CommitPipe::discard(const InstBundle* bundle) {
    if (!bundle) {
        return;
    }
    for (auto* inst : *bundle) {
        core_.discardRenamed(inst);
    }
}

void CommitPipe::completeRedirectFlush() {
    core_.completeRedirectFlush();
}

void CommitPipe::traceRetired(coropulse::TickId tick, const RetiredInstTrace& inst) {
    if (!trace_out_ || (trace_limit_ != 0 && trace_count_ >= trace_limit_)) {
        return;
    }

    auto& os = *trace_out_;
    os << "trace tick=" << tick
       << " seq=" << inst.sequence
       << " pc=0x" << std::hex << inst.pc
       << " bits=0x" << std::setw(8) << std::setfill('0') << inst.bits
       << std::dec << std::setfill(' ')
       << " op=" << opcodeName(inst.opcode);
    if (inst.writes_rd) {
        os << " x" << inst.rd << "=0x" << std::hex << inst.value << std::dec;
    }
    if (inst.store) {
        os << " store" << inst.store->bytes
           << " [0x" << std::hex << inst.store->address << "]=0x"
           << inst.store->value << std::dec;
    }
    if (inst.redirect) {
        os << " redirect=0x" << std::hex << inst.redirect->next_pc << std::dec;
        if (inst.redirect->halt) {
            os << " halt";
        }
    }
    os << '\n';
    ++trace_count_;
}

} // namespace riscv_cpu
