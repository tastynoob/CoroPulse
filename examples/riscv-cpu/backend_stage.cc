#include "stages.hh"

#include <utility>

namespace riscv_cpu {
namespace {

std::size_t countPhysicalDestinations(const InstBundle& bundle) {
    std::size_t count = 0;
    for (const auto* inst : bundle) {
        if (writesRd(inst->staticInst()) && inst->staticInst().rd != 0) {
            ++count;
        }
    }
    return count;
}

struct IssuedBundles {
    InstBundle execute;
    InstBundle memory;
};

IssuedBundles splitIssued(InstBundle&& issued) {
    IssuedBundles result;
    result.execute.reserve(issued.size());
    result.memory.reserve(issued.size());
    for (auto* inst : issued) {
        if (isMemory(inst->staticInst())) {
            result.memory.push_back(inst);
        } else {
            result.execute.push_back(inst);
        }
    }
    return result;
}

} // namespace

BackendStage::BackendStage(CoreState& core, SimpleSram& sram,
                           std::size_t rename_width,
                           std::size_t frontend_queue_capacity,
                           std::size_t fetch_width,
                           std::size_t decode_width,
                           std::size_t issue_capacity,
                           std::size_t load_queue_capacity,
                           std::size_t store_queue_capacity,
                           std::size_t dispatch_width,
                           std::size_t issue_width,
                           std::size_t memory_width,
                           std::size_t commit_width,
                           std::ostream* trace_out,
                           std::size_t trace_limit)
    : core_(core),
      backend_(core, sram, rename_width, frontend_queue_capacity, fetch_width,
               decode_width, issue_capacity, load_queue_capacity,
               store_queue_capacity, dispatch_width, issue_width, memory_width,
               commit_width, trace_out, trace_limit) {}

coropulse::Task<void> BackendStage::process() {
    for (;; co_yield coropulse::tickDone{}) {
        const bool flushing = backend_.commit.beginTick();

        if (backend_.commit.hasPendingRedirect()) {
            if (!backend_.commit.publishPendingRedirect(redirect_out, stats)) {
                can_accept.set(false);
                continue;
            }
        }

        if (flushing) {
            backend_.flushAfterRedirect(nullptr, in);
            continue;
        }

        if (!backend_.commit.publishPredictorUpdates(predictor_update_out)) {
            can_accept.set(false);
            continue;
        }

        auto completion = backend_.execute.collectCompletions(currentTick(), stats);
        auto memory_completion =
            backend_.memory.collectCompletions(currentTick(), stats);
        completion.insert(completion.end(), memory_completion.begin(),
                          memory_completion.end());

        if (!completion.empty()) {
            backend_.issue.rememberCompletions(completion);
            backend_.commit.markCompleted(completion);
        }

        auto result = backend_.commit.retire(currentTick(), stats);
        backend_.commit.queuePredictorUpdates(std::move(result.branch_updates));
        (void)backend_.commit.publishPredictorUpdates(predictor_update_out);
        if (result.redirect) {
            if (backend_.rename.hasBundle()) {
                const auto& bundle = backend_.rename.bundle();
                backend_.commit.discard(&bundle);
                backend_.rename.clear();
            }
            (void)backend_.commit.publishPendingRedirect(redirect_out, stats);
            can_accept.set(false);
            continue;
        }

        if (backend_.rename.hasBundle()) {
            auto dispatch = backend_.rename.takeBundle();
            backend_.commit.dispatch(dispatch);
            backend_.lsq.acceptRenamed(dispatch, stats);
            backend_.issue.acceptRenamed(dispatch, stats);
        }

        auto issued = backend_.issue.issue(stats);
        if (!issued.empty()) {
            auto split = splitIssued(std::move(issued));
            if (!split.execute.empty()) {
                backend_.execute.accept(std::move(split.execute), currentTick(), stats);
            }
            if (!split.memory.empty()) {
                stats.memory_issued += split.memory.size();
                backend_.memory.accept(std::move(split.memory), currentTick(), stats);
            }
        }

        bool phys_ready = true;
        bool rob_ready = true;
        bool issue_ready = true;
        bool lsq_ready = true;
        if (backend_.decode.hasBundle()) {
            const auto& bundle = backend_.decode.bundle();
            phys_ready = core_.canAllocatePhysicalRegisters(
                countPhysicalDestinations(bundle));
            rob_ready = core_.canAllocateRob(bundle.size());
            issue_ready = backend_.issue.canAccept(bundle.size());
            lsq_ready = backend_.lsq.canAccept(bundle);

            if (!issue_ready) {
                ++stats.issue_backpressure_stalls;
            }
            if (!phys_ready) {
                ++stats.resource_stalls;
            }
            if (!rob_ready) {
                ++stats.rob_resource_stalls;
            }
            if (!lsq_ready) {
                ++stats.lsq_resource_stalls;
            }
        }

        const bool rename_ready = backend_.rename.slotOpen() && phys_ready &&
                                  rob_ready && issue_ready && lsq_ready;
        if (rename_ready && backend_.decode.hasBundle()) {
            backend_.rename.accept(core_, backend_.decode.takeBundle(), stats);
        }

        const bool decode_ready = backend_.decode.slotOpen();
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
