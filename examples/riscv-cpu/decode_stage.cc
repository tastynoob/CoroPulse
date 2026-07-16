#include "stages.hh"

#include <utility>

namespace riscv_cpu {

coropulse::Task<void> DecodeStage::process() {
    for (;; co_yield coropulse::tickDone{}) {
        const bool flushing = redirect_in.read().has_value();
        if (flushing) {
            (void)in.take();
            continue;
        }

        const bool rename_ready = co_await rename_can_accept.read();
        const bool input_ready = rename_ready && out.canWrite();
        can_accept.set(input_ready);

        if (!input_ready && in.hasValue()) {
            ++backpressure_stalls_;
        }

        if (input_ready) {
            if (auto fetched = in.take()) {
                decoded_ += fetched->size();
                (void)out.write(std::move(*fetched));
            }
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
