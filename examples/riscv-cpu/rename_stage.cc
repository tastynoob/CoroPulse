#include "stages.hh"

#include <stdexcept>
#include <utility>

namespace riscv_cpu {

RenameStage::RenameStage(CoreState& core, std::size_t rename_width)
    : core_(core),
      rename_width_(rename_width) {
    if (rename_width_ == 0) {
        throw std::runtime_error("rename width must be greater than zero");
    }
}

coropulse::Task<void> RenameStage::process() {
    for (;; co_yield coropulse::tickDone{}) {
        const bool flushing = redirect_in.read().has_value();
        if (flushing) {
            (void)in.take();
            core_.completeRedirectFlush();
            continue;
        }

        const bool issue_ready = co_await issue_can_accept.read();
        const bool resource_ready = core_.freePhysicalRegisters() >= rename_width_;
        const bool input_ready = issue_ready && resource_ready && out.canWrite();
        can_accept.set(input_ready);

        if (!issue_ready && in.hasValue()) {
            ++issue_backpressure_stalls_;
        }
        if (!resource_ready && in.hasValue()) {
            ++resource_stalls_;
        }

        if (input_ready) {
            if (auto decoded = in.take()) {
                for (auto* inst : *decoded) {
                    core_.rename(inst);
                }
                renamed_ += decoded->size();
                (void)out.write(std::move(*decoded));
            }
        }
    }
    co_return;
}

std::size_t RenameStage::renamedCount() const {
    return renamed_;
}

std::size_t RenameStage::resourceStalls() const {
    return resource_stalls_;
}

std::size_t RenameStage::issueBackpressureStalls() const {
    return issue_backpressure_stalls_;
}

} // namespace riscv_cpu
