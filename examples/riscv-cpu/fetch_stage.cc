#include "stages.hh"

namespace riscv_cpu {

FetchStage::FetchStage(const SimpleSram& sram, DynInstPool& inst_pool)
    : sram_(sram),
      inst_pool_(inst_pool) {}

coropulse::Task<void> FetchStage::process() {
    for (;; co_yield coropulse::tickDone{}) {
        if (auto redirect = redirect_in.read()) {
            applyRedirect(*redirect);
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

        if (!pending_ && pc_ / 4 < sram_.instructionCount()) {
            const auto fetch_pc = pc_;
            const auto predicted_next_pc = fetch_pc + 4;
            pending_ = inst_pool_.create(sram_.loadInstruction(fetch_pc), fetch_pc,
                                         predicted_next_pc);
            pc_ = predicted_next_pc;
            ++fetched_;
        } else if (!pending_) {
            halted_ = true;
        }

        if (pending_ && out.write(pending_)) {
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

void FetchStage::applyRedirect(const ControlRedirect& redirect) {
    ++redirects_;
    pending_ = nullptr;
    pc_ = redirect.next_pc;
    halted_ = redirect.halt;
}

} // namespace riscv_cpu
