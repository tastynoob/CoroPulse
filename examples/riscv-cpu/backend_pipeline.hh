#pragma once

#include "core_state.hh"
#include "inst.hh"
#include "memory.hh"
#include "sim.hh"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <iosfwd>
#include <optional>
#include <vector>

namespace riscv_cpu {

struct BackendStats {
    std::size_t decoded = 0;
    std::size_t decode_backpressure_stalls = 0;
    std::size_t renamed = 0;
    std::size_t resource_stalls = 0;
    std::size_t issue_backpressure_stalls = 0;
    std::size_t accepted = 0;
    std::size_t issued = 0;
    std::size_t execute_accepted = 0;
    std::size_t execute_completed = 0;
    std::size_t retired = 0;
    std::size_t redirects = 0;
    std::size_t frontend_queue_max_occupancy = 0;
    std::size_t frontend_queue_stalls = 0;
};

class FrontendQueue {
public:
    FrontendQueue(std::size_t capacity, std::size_t fetch_width,
                  std::size_t decode_width);

    bool canAcceptFetch() const;
    bool hasDecodeBundle() const noexcept;
    void acceptFetch(InstBundle&& bundle, BackendStats& stats);
    InstBundle popDecodeBundle();
    void clear() noexcept;

private:
    std::size_t capacity_;
    std::size_t fetch_width_;
    std::size_t decode_width_;
    std::deque<DynInstPtr> queue_;
};

class DecodePipe {
public:
    bool slotOpen() const noexcept;
    bool hasBundle() const noexcept;
    InstBundle takeBundle();
    void accept(InstBundle&& bundle, BackendStats& stats);
    void clear() noexcept;

private:
    std::optional<InstBundle> reg_;
};

class RenamePipe {
public:
    bool slotOpen() const noexcept;
    bool hasIssueBundle() const noexcept;
    const InstBundle& issueBundle() const;
    const InstBundle* readCommitDispatch();
    void markIssueConsumed();
    void clearIfConsumed();
    void accept(CoreState& core, InstBundle&& bundle, BackendStats& stats);
    void clear() noexcept;

private:
    std::optional<InstBundle> reg_;
    bool issue_consumed_ = false;
    bool commit_consumed_ = false;
};

class IssuePipe {
public:
    IssuePipe(CoreState& core, std::size_t capacity, std::size_t dispatch_width,
              std::size_t issue_width);

    bool canAcceptDispatch() const;
    void rememberCompletions(const InstBundle& completions);
    void acceptRenamed(const InstBundle& bundle, BackendStats& stats);
    InstBundle issue(BackendStats& stats);
    void clear();

private:
    struct Entry {
        DynInstPtr inst = nullptr;
    };

    bool operandsReady(const Entry& entry) const;
    bool canIssue(const Entry& entry) const;
    std::vector<std::size_t> findReadyBundle() const;
    Entry makeEntry(DynInstPtr inst) const;
    void rememberCompletion(DynInstPtr completion);
    void applyWakeup(DynInstPtr completion);
    void applyKnownWakeups(Entry& entry) const;
    void applyKnownWakeup(Operand& operand) const;
    static void wakeOperand(Operand& operand, DynInstPtr producer);

    CoreState& core_;
    std::size_t capacity_;
    std::size_t dispatch_width_;
    std::size_t issue_width_;
    std::vector<Entry> queue_;
    std::vector<char> completed_;
};

class ExecutePipe {
public:
    ExecutePipe(CoreState& core, SimpleSram& sram);

    void accept(InstBundle&& bundle, coropulse::TickId tick, BackendStats& stats);
    InstBundle collectCompletions(coropulse::TickId tick, BackendStats& stats);
    void clear();

private:
    struct Executing {
        DynInstPtr inst = nullptr;
        coropulse::TickId done_tick = 0;
    };

    void execute(DynInstPtr inst, coropulse::TickId tick);
    void completeReadyUops(coropulse::TickId tick);

    CoreState& core_;
    SimpleSram& sram_;
    std::vector<Executing> executing_;
    InstBundle pending_completion_;
};

class CommitPipe {
public:
    CommitPipe(CoreState& core, SimpleSram& sram, std::size_t commit_width,
               std::ostream* trace_out, std::size_t trace_limit);

    bool beginTick();
    bool hasPendingRedirect() const noexcept;
    bool publishPendingRedirect(coropulse::Output<ControlRedirect>& output,
                                BackendStats& stats);
    bool publishPredictorUpdates(coropulse::Output<BranchUpdateBundle>& output);
    void queuePredictorUpdates(BranchUpdateBundle&& updates);
    RetireResult retire(coropulse::TickId tick, BackendStats& stats);
    void markCompleted(const InstBundle& bundle);
    void dispatch(const InstBundle& bundle);
    void discard(const InstBundle* bundle);
    void completeRedirectFlush();

private:
    void traceRetired(coropulse::TickId tick, const RetiredInstTrace& inst);

    CoreState& core_;
    SimpleSram& sram_;
    std::size_t commit_width_;
    std::ostream* trace_out_ = nullptr;
    std::size_t trace_limit_ = 0;
    std::size_t trace_count_ = 0;
    std::optional<ControlRedirect> pending_redirect_;
    BranchUpdateBundle pending_updates_;
    bool flush_next_tick_ = false;
};

class BackendPipeline {
public:
    BackendPipeline(CoreState& core, SimpleSram& sram, std::size_t rename_width,
                    std::size_t frontend_queue_capacity,
                    std::size_t fetch_width, std::size_t decode_width,
                    std::size_t issue_capacity, std::size_t dispatch_width,
                    std::size_t issue_width, std::size_t commit_width,
                    std::ostream* trace_out, std::size_t trace_limit);

    std::size_t renameWidth() const noexcept;
    void flushAfterRedirect(const InstBundle* dispatch,
                            coropulse::Input<InstBundle>& input);

    FrontendQueue frontend_queue;
    DecodePipe decode;
    RenamePipe rename;
    IssuePipe issue;
    ExecutePipe execute;
    CommitPipe commit;

private:
    std::size_t rename_width_;
};

} // namespace riscv_cpu
