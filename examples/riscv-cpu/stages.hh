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

class FetchStage final : public coropulse::Component {
public:
    coropulse::Input<ControlRedirect> redirect_in{"commit_redirect"};
    coropulse::SignalInput<bool> fifo_can_accept{"fetch_decode_fifo_can_accept"};
    coropulse::Output<InstBundle> out{"fetch_to_fifo"};

    FetchStage(const SimpleSram& sram, DynInstPool& inst_pool, std::size_t fetch_width);
    coropulse::Task<void> process() override;

    std::size_t fetchedCount() const;
    std::size_t backpressureStalls() const;
    std::size_t controlStalls() const;
    std::size_t redirectCount() const;
    bool halted() const noexcept;
    bool architecturalHalted() const noexcept;

private:
    void applyRedirect(const ControlRedirect& redirect);

    const SimpleSram& sram_;
    DynInstPool& inst_pool_;
    std::size_t fetch_width_;
    std::uint64_t pc_ = 0;
    std::size_t fetched_ = 0;
    std::size_t backpressure_stalls_ = 0;
    std::size_t control_stalls_ = 0;
    std::size_t redirects_ = 0;
    bool halted_ = false;
    bool architectural_halted_ = false;
};

class FetchDecodeFifo final : public coropulse::Component {
public:
    coropulse::Input<ControlRedirect> redirect_in{"commit_redirect"};
    coropulse::Input<InstBundle> in{"fetch_to_fifo"};
    coropulse::SignalInput<bool> decode_can_accept{"decode_can_accept"};
    coropulse::SignalOutput<bool> can_accept{"fetch_decode_fifo_can_accept"};
    coropulse::Output<InstBundle> out{"fifo_to_decode"};

    FetchDecodeFifo(std::size_t capacity, std::size_t fetch_width,
                    std::size_t decode_width);
    coropulse::Task<void> process() override;

    std::size_t maxOccupancy() const;
    std::size_t overflowStalls() const;

private:
    bool canAcceptFetch() const;
    InstBundle popDecodeBundle();

    std::size_t capacity_;
    std::size_t fetch_width_;
    std::size_t decode_width_;
    std::deque<DynInstPtr> queue_;
    std::size_t max_occupancy_ = 0;
    std::size_t overflow_stalls_ = 0;
};

class DecodeStage final : public coropulse::Component {
public:
    coropulse::Input<ControlRedirect> redirect_in{"commit_redirect"};
    coropulse::Input<InstBundle> in{"fifo_to_decode"};
    coropulse::SignalInput<bool> rename_can_accept{"rename_can_accept"};
    coropulse::SignalOutput<bool> can_accept{"decode_can_accept"};
    coropulse::Output<InstBundle> out{"decode_to_rename"};

    coropulse::Task<void> process() override;

    std::size_t decodedCount() const;
    std::size_t backpressureStalls() const;

private:
    std::size_t decoded_ = 0;
    std::size_t backpressure_stalls_ = 0;
};

class RenameStage final : public coropulse::Component {
public:
    coropulse::Input<ControlRedirect> redirect_in{"commit_redirect"};
    coropulse::Input<InstBundle> in{"decode_to_rename"};
    coropulse::SignalInput<bool> issue_can_accept{"issue_can_accept"};
    coropulse::SignalOutput<bool> can_accept{"rename_can_accept"};
    coropulse::Output<InstBundle> out{"rename_to_issue"};

    RenameStage(CoreState& core, std::size_t rename_width);
    coropulse::Task<void> process() override;

    std::size_t renamedCount() const;
    std::size_t resourceStalls() const;
    std::size_t issueBackpressureStalls() const;

private:
    CoreState& core_;
    std::size_t rename_width_;
    std::size_t renamed_ = 0;
    std::size_t resource_stalls_ = 0;
    std::size_t issue_backpressure_stalls_ = 0;
};

class IssueStage final : public coropulse::Component {
public:
    coropulse::Input<ControlRedirect> redirect_in{"commit_redirect"};
    coropulse::Input<InstBundle> rename_in{"rename_to_issue"};
    coropulse::Input<ExecResultBundle> completion_in{"execute_to_commit"};
    coropulse::SignalOutput<bool> can_accept{"issue_can_accept"};
    coropulse::Output<InstBundle> issue_out{"issue_to_execute"};

    IssueStage(CoreState& core, std::size_t capacity, std::size_t dispatch_width,
               std::size_t issue_width);
    coropulse::Task<void> process() override;

    std::size_t issuedCount() const;
    std::size_t acceptedCount() const;
    std::size_t outputStalls() const;

private:
    struct IssueEntry {
        DynInstPtr inst = nullptr;
        Operand src1;
        Operand src2;
        bool memory = false;
    };

    bool operandsReady(const IssueEntry& entry) const;
    bool canIssue(const IssueEntry& entry) const;
    std::vector<std::size_t> findReadyBundle() const;
    IssueEntry makeEntry(DynInstPtr inst) const;
    void rememberCompletion(const ExecResult& completion);
    void rememberCompletions(const ExecResultBundle& completions);
    void applyWakeup(const ExecResult& completion);
    void applyKnownWakeups(IssueEntry& entry) const;
    void applyKnownWakeup(Operand& operand) const;
    void wakeOperand(Operand& operand, const ExecResult& completion);

    CoreState& core_;
    std::size_t capacity_;
    std::size_t dispatch_width_;
    std::size_t issue_width_;
    std::vector<IssueEntry> queue_;
    std::vector<std::optional<ExecResult>> completed_;
    std::size_t accepted_ = 0;
    std::size_t issued_ = 0;
    std::size_t output_stalls_ = 0;
};

class ExecuteStage final : public coropulse::Component {
public:
    coropulse::Input<ControlRedirect> redirect_in{"commit_redirect"};
    coropulse::Input<InstBundle> issue_in{"issue_to_execute"};
    coropulse::Output<ExecResultBundle> completion_out{"execute_to_commit"};

    explicit ExecuteStage(SimpleSram& sram);
    coropulse::Task<void> process() override;

    std::size_t acceptedCount() const;
    std::size_t completedCount() const;

private:
    struct Executing {
        DynInstPtr inst = nullptr;
        coropulse::TickId done_tick = 0;
    };

    void publishCompletion();
    void completeReadyUops();
    ExecResult execute(DynInstPtr inst);

    SimpleSram& sram_;
    std::vector<Executing> executing_;
    ExecResultBundle pending_completion_;
    std::size_t accepted_ = 0;
    std::size_t completed_ = 0;
};

class CommitStage final : public coropulse::Component {
public:
    coropulse::Input<InstBundle> dispatch_in{"rename_to_issue"};
    coropulse::Input<ExecResultBundle> completion_in{"execute_to_commit"};
    coropulse::Output<ControlRedirect> redirect_out{"commit_redirect"};

    CommitStage(CoreState& core, SimpleSram& sram, std::size_t commit_width,
                std::ostream* trace_out = nullptr, std::size_t trace_limit = 0);
    coropulse::Task<void> process() override;

    std::size_t retiredCount() const;
    std::size_t redirectCount() const;

private:
    void traceRetired(const RetiredInstTrace& inst);

    CoreState& core_;
    SimpleSram& sram_;
    std::size_t commit_width_;
    std::ostream* trace_out_ = nullptr;
    std::size_t trace_limit_ = 0;
    std::size_t trace_count_ = 0;
    std::optional<ControlRedirect> pending_redirect_;
    std::size_t retired_ = 0;
    std::size_t redirects_ = 0;
    bool flush_next_tick_ = false;
};

} // namespace riscv_cpu
