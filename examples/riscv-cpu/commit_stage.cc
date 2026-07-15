#include "stages.hh"

namespace riscv_cpu {

CommitStage::CommitStage(CoreState& core, std::size_t commit_width)
    : core_(core),
      commit_width_(commit_width) {}

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
                if (dispatch) {
                    core_.discardRenamed(*dispatch);
                }
                flush_next_tick_ = true;
                continue;
            }
        }

        if (flushing) {
            if (dispatch) {
                core_.discardRenamed(*dispatch);
            }
            continue;
        }

        if (completion) {
            core_.markCompleted(*completion);
        }

        const auto result = core_.retire(commit_width_);
        retired_ += result.retired;
        if (result.redirect) {
            if (dispatch) {
                core_.discardRenamed(*dispatch);
            }

            pending_redirect_ = result.redirect;
            if (redirect_out.write(*pending_redirect_)) {
                pending_redirect_.reset();
                flush_next_tick_ = true;
                ++redirects_;
            }
            continue;
        }

        if (dispatch) {
            core_.dispatchRenamed(*dispatch);
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

} // namespace riscv_cpu
