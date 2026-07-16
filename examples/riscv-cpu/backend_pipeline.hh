#pragma once

#include "core_state.hh"
#include "inst.hh"
#include "memory.hh"
#include "sim.hh"

#include <cstddef>
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
    std::size_t output_stalls = 0;
    std::size_t retired = 0;
    std::size_t redirects = 0;
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
    void rememberCompletions(const ExecResultBundle& completions);
    void acceptRenamed(const InstBundle& bundle, BackendStats& stats);
    void issue(coropulse::Output<InstBundle>& output, BackendStats& stats);
    void clear();

private:
    struct Entry {
        DynInstPtr inst = nullptr;
        Operand src1;
        Operand src2;
        bool memory = false;
    };

    bool operandsReady(const Entry& entry) const;
    bool canIssue(const Entry& entry) const;
    std::vector<std::size_t> findReadyBundle() const;
    Entry makeEntry(DynInstPtr inst) const;
    void rememberCompletion(const ExecResult& completion);
    void applyWakeup(const ExecResult& completion);
    void applyKnownWakeups(Entry& entry) const;
    void applyKnownWakeup(Operand& operand) const;
    static void wakeOperand(Operand& operand, const ExecResult& completion);

    CoreState& core_;
    std::size_t capacity_;
    std::size_t dispatch_width_;
    std::size_t issue_width_;
    std::vector<Entry> queue_;
    std::vector<std::optional<ExecResult>> completed_;
};

class CommitPipe {
public:
    CommitPipe(CoreState& core, SimpleSram& sram, std::size_t commit_width,
               std::ostream* trace_out, std::size_t trace_limit);

    bool beginTick();
    bool hasPendingRedirect() const noexcept;
    bool publishPendingRedirect(coropulse::Output<ControlRedirect>& output,
                                BackendStats& stats);
    RetireResult retire(coropulse::TickId tick, BackendStats& stats);
    void markCompleted(const ExecResultBundle& bundle);
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
    bool flush_next_tick_ = false;
};

class BackendPipeline {
public:
    BackendPipeline(CoreState& core, SimpleSram& sram, std::size_t rename_width,
                    std::size_t issue_capacity, std::size_t dispatch_width,
                    std::size_t issue_width, std::size_t commit_width,
                    std::ostream* trace_out, std::size_t trace_limit);

    std::size_t renameWidth() const noexcept;
    void flushAfterRedirect(const InstBundle* dispatch,
                            coropulse::Input<InstBundle>& input);

    DecodePipe decode;
    RenamePipe rename;
    IssuePipe issue;
    CommitPipe commit;

private:
    std::size_t rename_width_;
};

} // namespace riscv_cpu
