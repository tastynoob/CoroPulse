#include "stages.hh"

namespace riscv_cpu {

RenameStage::RenameStage(CoreState& core) : core_(core) {}

coropulse::Task<void> RenameStage::process() {
    for (;; co_yield coropulse::tickDone{}) {
        const bool issue_ready = co_await issue_can_accept.read();
        const bool resource_ready = pending_ || core_.canRenameAny();
        const bool input_ready = !pending_ && issue_ready && resource_ready;
        can_accept.set(input_ready);

        if (!issue_ready && (pending_ || in.hasValue())) {
            ++issue_backpressure_stalls_;
        }
        if (!resource_ready) {
            ++resource_stalls_;
        }

        if (!pending_ && input_ready) {
            if (auto decoded = in.read()) {
                pending_ = *decoded;
                core_.rename(pending_);
                ++renamed_;
            }
        }

        if (pending_ && issue_ready && out.write(pending_)) {
            pending_ = nullptr;
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
