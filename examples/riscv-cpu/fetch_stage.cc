#include "stages.hh"

namespace riscv_cpu {

FetchStage::FetchStage(const SimpleSram& sram, DynInstPool& inst_pool)
    : sram_(sram),
      inst_pool_(inst_pool) {}

coropulse::Task<void> FetchStage::process() {
    for (;; co_yield coropulse::tickDone{}) {
        if (auto redirect = redirect_in.read()) {
            ++redirects_;
            waiting_for_redirect_ = false;
            pc_ = redirect->next_pc;
            halted_ = redirect->halt;
        }

        const bool decode_ready = co_await decode_can_accept.read();
        if (halted_) {
            continue;
        }
        if (!decode_ready) {
            if (pending_ || pc_ / 4 < sram_.instructionCount()) {
                ++backpressure_stalls_;
            }
            continue;
        }
        if (waiting_for_redirect_) {
            ++control_stalls_;
            continue;
        }

        if (!pending_ && pc_ / 4 < sram_.instructionCount()) {
            pending_ = inst_pool_.create(sram_.loadInstruction(pc_), pc_);
            pc_ += 4;
            ++fetched_;
        } else if (!pending_) {
            halted_ = true;
        }

        if (pending_ && out.write(pending_)) {
            if (isControlFlow(pending_->staticInst())) {
                waiting_for_redirect_ = true;
            }
            pending_ = nullptr;
        }
    }
    co_return;
}

std::size_t FetchStage::fetchedCount() const {
    return fetched_;
}

std::size_t FetchStage::backpressureStalls() const {
    return backpressure_stalls_;
}

std::size_t FetchStage::controlStalls() const {
    return control_stalls_;
}

std::size_t FetchStage::redirectCount() const {
    return redirects_;
}

bool FetchStage::halted() const noexcept {
    return halted_;
}

} // namespace riscv_cpu
