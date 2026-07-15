#include "stages.hh"

namespace riscv_cpu {

coropulse::Task<void> DecodeStage::process() {
    for (;; co_yield coropulse::tickDone{}) {
        const bool rename_ready = co_await rename_can_accept.read();
        const bool input_ready = !pending_ && rename_ready;
        can_accept.set(input_ready);

        if (!rename_ready && (pending_ || in.hasValue())) {
            ++backpressure_stalls_;
        }

        if (!pending_ && input_ready) {
            if (auto fetched = in.read()) {
                pending_ = *fetched;
                ++decoded_;
            }
        }

        if (pending_ && rename_ready && out.write(pending_)) {
            pending_ = nullptr;
        }
    }
    co_return;
}

std::size_t DecodeStage::decodedCount() const {
    return decoded_;
}

std::size_t DecodeStage::backpressureStalls() const {
    return backpressure_stalls_;
}

} // namespace riscv_cpu
