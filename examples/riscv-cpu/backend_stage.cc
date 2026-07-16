#include "stages.hh"

#include <utility>

namespace riscv_cpu {

BackendStage::BackendStage(CoreState& core, SimpleSram& sram,
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
    : core_(core),
      backend_(core, sram, rename_width, frontend_queue_capacity, fetch_width,
               decode_width, issue_capacity, dispatch_width, issue_width,
               commit_width, trace_out, trace_limit) {}

coropulse::Task<void> BackendStage::process() {
    for (;; co_yield coropulse::tickDone{}) {
        const bool flushing = backend_.commit.beginTick();
        const bool decode_slot_open = backend_.decode.slotOpen();
        const bool rename_slot_open = backend_.rename.slotOpen();

        if (backend_.commit.hasPendingRedirect()) {
            const auto* dispatch = backend_.rename.readCommitDispatch();
            if (!backend_.commit.publishPendingRedirect(redirect_out, stats)) {
                backend_.commit.discard(dispatch);
                can_accept.set(false);
                backend_.rename.clearIfConsumed();
                continue;
            }
        }

        if (flushing) {
            const auto* dispatch = backend_.rename.readCommitDispatch();
            backend_.flushAfterRedirect(dispatch, in);
            continue;
        }

        if (!backend_.commit.publishPredictorUpdates(predictor_update_out)) {
            can_accept.set(false);
            backend_.rename.clearIfConsumed();
            continue;
        }

        const auto completion = backend_.execute.collectCompletions(currentTick(), stats);
        const auto* dispatch = backend_.rename.readCommitDispatch();

        if (!completion.empty()) {
            backend_.issue.rememberCompletions(completion);
            backend_.commit.markCompleted(completion);
        }

        auto result = backend_.commit.retire(currentTick(), stats);
        backend_.commit.queuePredictorUpdates(std::move(result.branch_updates));
        (void)backend_.commit.publishPredictorUpdates(predictor_update_out);
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

        auto issued = backend_.issue.issue(stats);
        if (!issued.empty()) {
            backend_.execute.accept(std::move(issued), currentTick(), stats);
        }

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

        if (!issue_ready && backend_.decode.hasBundle()) {
            ++stats.issue_backpressure_stalls;
        }
        if (!resource_ready && backend_.decode.hasBundle()) {
            ++stats.resource_stalls;
        }

        if (rename_ready && backend_.decode.hasBundle()) {
            backend_.rename.accept(core_, backend_.decode.takeBundle(), stats);
        }

        if (!decode_ready && backend_.frontend_queue.hasDecodeBundle()) {
            ++stats.decode_backpressure_stalls;
        }
        if (decode_ready) {
            if (backend_.frontend_queue.hasDecodeBundle()) {
                backend_.decode.accept(backend_.frontend_queue.popDecodeBundle(), stats);
            }
        }

        const bool frontend_ready = backend_.frontend_queue.canAcceptFetch();
        can_accept.set(frontend_ready);
        if (!frontend_ready && in.hasValue()) {
            ++stats.frontend_queue_stalls;
        }
        if (frontend_ready) {
            if (auto fetched = in.take()) {
                backend_.frontend_queue.acceptFetch(std::move(*fetched), stats);
            }
        }
    }
    co_return;
}

} // namespace riscv_cpu
