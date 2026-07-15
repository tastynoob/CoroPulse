#pragma once

#include "core_state.hh"
#include "inst.hh"
#include "memory.hh"
#include "sim.hh"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <vector>

namespace riscv_cpu {

class FetchStage final : public coropulse::Component {
public:
    coropulse::Input<ControlRedirect> redirect_in{"commit_redirect"};
    coropulse::SignalInput<bool> decode_can_accept{"decode_can_accept"};
    coropulse::Output<DynInstPtr> out{"fetch_to_decode"};

    FetchStage(const SimpleSram& sram, DynInstPool& inst_pool);
    coropulse::Task<void> process() override;

    std::size_t fetchedCount() const;
    std::size_t backpressureStalls() const;
    std::size_t controlStalls() const;
    std::size_t redirectCount() const;
    bool halted() const noexcept;

private:
    void applyRedirect(const ControlRedirect& redirect);

    const SimpleSram& sram_;
    DynInstPool& inst_pool_;
    std::uint64_t pc_ = 0;
    std::size_t fetched_ = 0;
    std::size_t backpressure_stalls_ = 0;
    std::size_t control_stalls_ = 0;
    std::size_t redirects_ = 0;
    bool halted_ = false;
    DynInstPtr pending_ = nullptr;
};

class DecodeStage final : public coropulse::Component {
public:
    coropulse::Input<ControlRedirect> redirect_in{"commit_redirect"};
    coropulse::Input<DynInstPtr> in{"fetch_to_decode"};
    coropulse::SignalInput<bool> rename_can_accept{"rename_can_accept"};
    coropulse::SignalOutput<bool> can_accept{"decode_can_accept"};
    coropulse::Output<DynInstPtr> out{"decode_to_rename"};

    coropulse::Task<void> process() override;

    std::size_t decodedCount() const;
    std::size_t backpressureStalls() const;

private:
    std::size_t decoded_ = 0;
    std::size_t backpressure_stalls_ = 0;
    DynInstPtr pending_ = nullptr;
};

class RenameStage final : public coropulse::Component {
public:
    coropulse::Input<ControlRedirect> redirect_in{"commit_redirect"};
    coropulse::Input<DynInstPtr> in{"decode_to_rename"};
    coropulse::SignalInput<bool> issue_can_accept{"issue_can_accept"};
    coropulse::SignalOutput<bool> can_accept{"rename_can_accept"};
    coropulse::Output<DynInstPtr> out{"rename_to_issue"};

    explicit RenameStage(CoreState& core);
    coropulse::Task<void> process() override;

    std::size_t renamedCount() const;
    std::size_t resourceStalls() const;
    std::size_t issueBackpressureStalls() const;

private:
    CoreState& core_;
    std::size_t renamed_ = 0;
    std::size_t resource_stalls_ = 0;
    std::size_t issue_backpressure_stalls_ = 0;
    DynInstPtr pending_ = nullptr;
};

class IssueStage final : public coropulse::Component {
public:
    coropulse::Input<ControlRedirect> redirect_in{"commit_redirect"};
    coropulse::Input<DynInstPtr> rename_in{"rename_to_issue"};
    coropulse::Input<ExecResult> completion_in{"execute_to_commit"};
    coropulse::SignalOutput<bool> can_accept{"issue_can_accept"};
    coropulse::Output<DynInstPtr> issue_out{"issue_to_execute"};

    explicit IssueStage(std::size_t capacity);
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
    std::optional<std::size_t> findReady() const;
    IssueEntry makeEntry(DynInstPtr inst) const;
    void rememberCompletion(const ExecResult& completion);
    void applyWakeup(const ExecResult& completion);
    void applyKnownWakeups(IssueEntry& entry) const;
    void applyKnownWakeup(Operand& operand) const;
    void wakeOperand(Operand& operand, const ExecResult& completion);

    std::size_t capacity_;
    std::vector<IssueEntry> queue_;
    std::vector<std::optional<ExecResult>> completed_;
    std::size_t accepted_ = 0;
    std::size_t issued_ = 0;
    std::size_t output_stalls_ = 0;
};

class ExecuteStage final : public coropulse::Component {
public:
    coropulse::Input<ControlRedirect> redirect_in{"commit_redirect"};
    coropulse::Input<DynInstPtr> issue_in{"issue_to_execute"};
    coropulse::Output<ExecResult> completion_out{"execute_to_commit"};

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
    void completeReadyUop();
    ExecResult execute(DynInstPtr inst);

    SimpleSram& sram_;
    std::vector<Executing> executing_;
    std::optional<ExecResult> pending_completion_;
    std::size_t accepted_ = 0;
    std::size_t completed_ = 0;
};

class CommitStage final : public coropulse::Component {
public:
    coropulse::Input<DynInstPtr> dispatch_in{"rename_to_issue"};
    coropulse::Input<ExecResult> completion_in{"execute_to_commit"};
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
