#include "backend_pipeline.hh"

#include <algorithm>
#include <stdexcept>

namespace riscv_cpu {

FrontendQueue::FrontendQueue(std::size_t capacity, std::size_t fetch_width,
                             std::size_t decode_width)
    : capacity_(capacity),
      fetch_width_(fetch_width),
      decode_width_(decode_width) {
    if (capacity_ == 0) {
        throw std::runtime_error("frontend queue capacity must be greater than zero");
    }
    if (fetch_width_ == 0 || decode_width_ == 0) {
        throw std::runtime_error("fetch/decode widths must be greater than zero");
    }
    if (capacity_ < fetch_width_) {
        throw std::runtime_error("frontend queue capacity must cover fetch width");
    }
}

bool FrontendQueue::canAcceptFetch() const {
    return capacity_ >= queue_.size() + fetch_width_;
}

bool FrontendQueue::hasDecodeBundle() const noexcept {
    return !queue_.empty();
}

void FrontendQueue::acceptFetch(InstBundle&& bundle, BackendStats& stats) {
    if (!canAcceptFetch()) {
        throw std::runtime_error("frontend queue accepted fetch while full");
    }

    for (auto* inst : bundle) {
        queue_.push_back(inst);
    }
    if (queue_.size() > stats.frontend_queue_max_occupancy) {
        stats.frontend_queue_max_occupancy = queue_.size();
    }
}

InstBundle FrontendQueue::popDecodeBundle() {
    InstBundle bundle;
    const auto count = std::min(decode_width_, queue_.size());
    bundle.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        bundle.push_back(queue_.front());
        queue_.pop_front();
    }
    return bundle;
}

void FrontendQueue::clear() noexcept {
    queue_.clear();
}

BackendPipeline::BackendPipeline(CoreState& core, SimpleSram& sram,
                                 std::size_t rename_width,
                                 std::size_t frontend_queue_capacity,
                                 std::size_t fetch_width,
                                 std::size_t decode_width,
                                 std::size_t issue_capacity,
                                 std::size_t dispatch_width,
                                 std::size_t issue_width,
                                 std::size_t commit_width,
                                 std::ostream* trace_out,
                                 std::size_t trace_limit)
    : frontend_queue(frontend_queue_capacity, fetch_width, decode_width),
      issue(core, issue_capacity, dispatch_width, issue_width),
      execute(core, sram),
      commit(core, sram, commit_width, trace_out, trace_limit),
      rename_width_(rename_width) {
    if (rename_width_ == 0 || commit_width == 0) {
        throw std::runtime_error("backend rename and commit widths must be greater than zero");
    }
}

std::size_t BackendPipeline::renameWidth() const noexcept {
    return rename_width_;
}

void BackendPipeline::flushAfterRedirect(const InstBundle* dispatch,
                                         coropulse::Input<InstBundle>& input) {
    commit.discard(dispatch);
    (void)input.take();
    frontend_queue.clear();
    decode.clear();
    rename.clear();
    issue.clear();
    execute.clear();
    commit.completeRedirectFlush();
}

} // namespace riscv_cpu
