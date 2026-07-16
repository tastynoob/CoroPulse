#pragma once

#include "backend_pipeline.hh"
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
    struct Stats {
        std::size_t fetched = 0;
        std::size_t backpressure_stalls = 0;
        std::size_t control_stalls = 0;
        std::size_t redirects = 0;
    };

    coropulse::Input<ControlRedirect> redirect_in{"commit_redirect"};
    coropulse::SignalInput<bool> fifo_can_accept{"fetch_decode_fifo_can_accept"};
    coropulse::Output<InstBundle> out{"fetch_to_fifo"};
    Stats stats;

    FetchStage(const SimpleSram& sram, DynInstPool& inst_pool, std::size_t fetch_width);
    coropulse::Task<void> process() override;

    bool halted() const noexcept;
    bool architecturalHalted() const noexcept;

private:
    void applyRedirect(const ControlRedirect& redirect);

    const SimpleSram& sram_;
    DynInstPool& inst_pool_;
    std::size_t fetch_width_;
    std::uint64_t pc_ = 0;
    bool halted_ = false;
    bool architectural_halted_ = false;
};

class FetchDecodeFifo final : public coropulse::Component {
public:
    struct Stats {
        std::size_t max_occupancy = 0;
        std::size_t overflow_stalls = 0;
    };

    coropulse::Input<ControlRedirect> redirect_in{"commit_redirect"};
    coropulse::Input<InstBundle> in{"fetch_to_fifo"};
    coropulse::SignalInput<bool> backend_can_accept{"backend_can_accept"};
    coropulse::SignalOutput<bool> can_accept{"fetch_decode_fifo_can_accept"};
    coropulse::Output<InstBundle> out{"fifo_to_backend"};
    Stats stats;

    FetchDecodeFifo(std::size_t capacity, std::size_t fetch_width,
                    std::size_t decode_width);
    coropulse::Task<void> process() override;

private:
    bool canAcceptFetch() const;
    InstBundle popDecodeBundle();

    std::size_t capacity_;
    std::size_t fetch_width_;
    std::size_t decode_width_;
    std::deque<DynInstPtr> queue_;
};

class BackendStage final : public coropulse::Component {
public:
    coropulse::Input<InstBundle> in{"fifo_to_backend"};
    coropulse::Input<ExecResultBundle> completion_in{"execute_to_backend"};
    coropulse::SignalOutput<bool> can_accept{"backend_can_accept"};
    coropulse::Output<InstBundle> issue_out{"issue_to_execute"};
    coropulse::Output<ControlRedirect> redirect_out{"commit_redirect"};
    BackendStats stats;

    BackendStage(CoreState& core, SimpleSram& sram, std::size_t rename_width,
                 std::size_t issue_capacity, std::size_t dispatch_width,
                 std::size_t issue_width, std::size_t commit_width,
                 std::ostream* trace_out = nullptr, std::size_t trace_limit = 0);
    coropulse::Task<void> process() override;

private:
    CoreState& core_;
    BackendPipeline backend_;
};

class ExecuteStage final : public coropulse::Component {
public:
    struct Stats {
        std::size_t accepted = 0;
        std::size_t completed = 0;
    };

    coropulse::Input<ControlRedirect> redirect_in{"commit_redirect"};
    coropulse::Input<InstBundle> issue_in{"issue_to_execute"};
    coropulse::Output<ExecResultBundle> completion_out{"execute_to_backend"};
    Stats stats;

    explicit ExecuteStage(SimpleSram& sram);
    coropulse::Task<void> process() override;

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
};

} // namespace riscv_cpu
