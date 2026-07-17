#pragma once

#include "backend_pipeline.hh"
#include "branch_predictor.hh"
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
        std::size_t frontend_squashes = 0;
        BranchPredictor::Stats predictor;
    };

    coropulse::Input<ControlRedirect> redirect_in{"commit_redirect"};
    coropulse::Input<BranchUpdateBundle> predictor_update_in{"predictor_update"};
    coropulse::SignalInput<bool> backend_can_accept{"backend_can_accept"};
    coropulse::Output<InstBundle> out{"fetch_to_backend"};
    Stats stats;

    FetchStage(const SimpleSram& sram, DynInstPool& inst_pool, std::size_t fetch_width);
    coropulse::Task<void> process() override;

    bool halted() const noexcept;
    bool architecturalHalted() const noexcept;

private:
    struct FetchSlot {
        const StaticInst* inst = nullptr;
        std::uint64_t pc = 0;
        std::uint64_t predicted_next_pc = 0;
        BranchPrediction prediction;
    };

    struct FetchPacket {
        std::vector<FetchSlot> slots;
        std::uint64_t micro_next_pc = 0;
        std::uint64_t final_next_pc = 0;
        bool has_control = false;
        std::size_t control_index = 0;
        coropulse::TickId tage_ready_tick = 0;
        bool finalized = false;
        bool correction_applied = false;
    };

    void applyPredictorUpdates();
    void applyRedirect(const ControlRedirect& redirect);
    bool emitReadyPacket(bool fifo_ready);
    bool canBuildPacket() const;
    FetchPacket buildPacket(std::uint64_t pc);
    bool finalizePrediction(FetchPacket& packet);
    InstBundle makeInstBundle(const FetchPacket& packet);

    const SimpleSram& sram_;
    DynInstPool& inst_pool_;
    std::size_t fetch_width_;
    BranchPredictor predictor_;
    std::deque<FetchPacket> pipe_;
    std::uint64_t pc_ = 0;
    bool halted_ = false;
    bool architectural_halted_ = false;
};

class BackendStage final : public coropulse::Component {
public:
    coropulse::Input<InstBundle> in{"fetch_to_backend"};
    coropulse::SignalOutput<bool> can_accept{"backend_can_accept"};
    coropulse::Output<ControlRedirect> redirect_out{"commit_redirect"};
    coropulse::Output<BranchUpdateBundle> predictor_update_out{"predictor_update"};
    BackendStats stats;

    BackendStage(CoreState& core, SimpleSram& sram, std::size_t rename_width,
                 std::size_t frontend_queue_capacity, std::size_t fetch_width,
                 std::size_t decode_width,
                 std::size_t issue_capacity, std::size_t load_queue_capacity,
                 std::size_t store_queue_capacity, std::size_t dispatch_width,
                 std::size_t issue_width, std::size_t memory_width,
                 std::size_t commit_width,
                 std::ostream* trace_out = nullptr, std::size_t trace_limit = 0);
    coropulse::Task<void> process() override;

private:
    CoreState& core_;
    BackendPipeline backend_;
};

} // namespace riscv_cpu
