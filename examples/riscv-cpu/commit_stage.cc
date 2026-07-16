#include "stages.hh"

#include <iomanip>
#include <ostream>

namespace riscv_cpu {
namespace {

template <class BundleRef>
void discardBundle(CoreState& core, const BundleRef& bundle) {
    if (!bundle) {
        return;
    }
    for (auto* inst : *bundle) {
        core.discardRenamed(inst);
    }
}

void dispatchBundle(CoreState& core, const InstBundle& bundle) {
    for (auto* inst : bundle) {
        core.dispatchRenamed(inst);
    }
}

void markCompletedBundle(CoreState& core, const ExecResultBundle& bundle) {
    for (const auto& completion : bundle) {
        core.markCompleted(completion);
    }
}

} // namespace

CommitStage::CommitStage(CoreState& core, SimpleSram& sram, std::size_t commit_width,
                         std::ostream* trace_out, std::size_t trace_limit)
    : core_(core),
      sram_(sram),
      commit_width_(commit_width),
      trace_out_(trace_out),
      trace_limit_(trace_limit) {}

coropulse::Task<void> CommitStage::process() {
    for (;; co_yield coropulse::tickDone{}) {
        auto dispatch = dispatch_in.read();
        auto completion = completion_in.read();
        const bool flushing = flush_next_tick_;
        flush_next_tick_ = false;

        if (pending_redirect_) {
            if (redirect_out.write(*pending_redirect_)) {
                pending_redirect_.reset();
                flush_next_tick_ = true;
                ++redirects_;
            } else {
                discardBundle(core_, dispatch);
                flush_next_tick_ = true;
                continue;
            }
        }

        if (flushing) {
            discardBundle(core_, dispatch);
            continue;
        }

        if (completion) {
            markCompletedBundle(core_, *completion);
        }

        const auto result = core_.retire(commit_width_);
        for (const auto& inst : result.trace) {
            traceRetired(inst);
        }
        for (const auto& store : result.stores) {
            sram_.store(store.address, store.value, store.bytes);
        }
        retired_ += result.retired;
        if (result.redirect) {
            discardBundle(core_, dispatch);

            pending_redirect_ = result.redirect;
            if (redirect_out.write(*pending_redirect_)) {
                pending_redirect_.reset();
                flush_next_tick_ = true;
                ++redirects_;
            }
            continue;
        }

        if (dispatch) {
            dispatchBundle(core_, *dispatch);
        }
    }
    co_return;
}

std::size_t CommitStage::retiredCount() const {
    return retired_;
}

std::size_t CommitStage::redirectCount() const {
    return redirects_;
}

void CommitStage::traceRetired(const RetiredInstTrace& inst) {
    if (!trace_out_ || (trace_limit_ != 0 && trace_count_ >= trace_limit_)) {
        return;
    }

    auto& os = *trace_out_;
    os << "trace tick=" << currentTick()
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
