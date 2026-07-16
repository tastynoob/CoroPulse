#include "stages.hh"

#include <utility>

namespace riscv_cpu {

BackendStage::BackendStage(CoreState& core, SimpleSram& sram,
                           std::size_t rename_width,
                           std::size_t issue_capacity,
                           std::size_t dispatch_width,
                           std::size_t issue_width,
                           std::size_t commit_width,
                           std::ostream* trace_out,
                           std::size_t trace_limit)
    : core_(core),
      backend_(core, sram, rename_width, issue_capacity, dispatch_width,
               issue_width, commit_width, trace_out, trace_limit) {}

coropulse::Task<void> BackendStage::process() {
    for (;; co_yield coropulse::tickDone{}) {
        const bool flushing = backend_.commit.beginTick();
        const bool decode_slot_open = backend_.decode.slotOpen();
        const bool rename_slot_open = backend_.rename.slotOpen();

        auto completion = completion_in.read();
        const auto* dispatch = backend_.rename.readCommitDispatch();

        if (backend_.commit.hasPendingRedirect()) {
            if (!backend_.commit.publishPendingRedirect(redirect_out, stats)) {
                backend_.commit.discard(dispatch);
                can_accept.set(false);
                backend_.rename.clearIfConsumed();
                continue;
            }
        }

        if (flushing) {
            backend_.flushAfterRedirect(dispatch, in);
            continue;
        }

        if (completion) {
            backend_.issue.rememberCompletions(*completion);
            backend_.commit.markCompleted(*completion);
        }

        const auto result = backend_.commit.retire(currentTick(), stats);
        if (result.redirect) {
            backend_.commit.discard(dispatch);
            (void)backend_.commit.publishPendingRedirect(redirect_out, stats);
            can_accept.set(false);
            backend_.rename.clearIfConsumed();
            continue;
        }

        if (dispatch) {
            backend_.commit.dispatch(*dispatch);
        }

        backend_.issue.issue(issue_out, stats);

        const bool issue_ready = backend_.issue.canAcceptDispatch();
        if (issue_ready && backend_.rename.hasIssueBundle()) {
            backend_.issue.acceptRenamed(backend_.rename.issueBundle(), stats);
            backend_.rename.markIssueConsumed();
        }
        backend_.rename.clearIfConsumed();

        const bool resource_ready =
            core_.freePhysicalRegisters() >= backend_.renameWidth();
        const bool rename_ready = issue_ready && resource_ready && rename_slot_open;
        const bool decode_ready = rename_ready && decode_slot_open;
        can_accept.set(decode_ready);

        if (!issue_ready && backend_.decode.hasBundle()) {
            ++stats.issue_backpressure_stalls;
        }
        if (!resource_ready && backend_.decode.hasBundle()) {
            ++stats.resource_stalls;
        }

        if (rename_ready && backend_.decode.hasBundle()) {
            backend_.rename.accept(core_, backend_.decode.takeBundle(), stats);
        }

        if (!decode_ready && in.hasValue()) {
            ++stats.decode_backpressure_stalls;
        }
        if (decode_ready) {
            if (auto fetched = in.take()) {
                backend_.decode.accept(std::move(*fetched), stats);
            }
        }
    }
    co_return;
}

} // namespace riscv_cpu
