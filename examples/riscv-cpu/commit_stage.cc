#include "stages.hh"

namespace riscv_cpu {

CommitStage::CommitStage(CoreState& core, std::size_t commit_width)
    : core_(core),
      commit_width_(commit_width) {}

coropulse::Task<void> CommitStage::process() {
    for (;; co_yield coropulse::tickDone{}) {
        if (auto completion = completion_in.read()) {
            core_.markCompleted(*completion);
        }

        retired_ += core_.retire(commit_width_);
    }
    co_return;
}

std::size_t CommitStage::retiredCount() const {
    return retired_;
}

} // namespace riscv_cpu
