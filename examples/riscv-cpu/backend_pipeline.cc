#include "backend_pipeline.hh"

#include <stdexcept>

namespace riscv_cpu {

BackendPipeline::BackendPipeline(CoreState& core, SimpleSram& sram,
                                 std::size_t rename_width,
                                 std::size_t issue_capacity,
                                 std::size_t dispatch_width,
                                 std::size_t issue_width,
                                 std::size_t commit_width,
                                 std::ostream* trace_out,
                                 std::size_t trace_limit)
    : issue(core, issue_capacity, dispatch_width, issue_width),
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
    decode.clear();
    rename.clear();
    issue.clear();
    commit.completeRedirectFlush();
}

} // namespace riscv_cpu
