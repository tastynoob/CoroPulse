#include "inst.hh"

namespace riscv_cpu {

DynInst::DynInst(const StaticInst& static_inst, std::uint64_t pc,
                 std::uint64_t predicted_next_pc,
                 const BranchPrediction& branch_prediction,
                 std::uint64_t fetch_sequence)
    : static_inst_(&static_inst),
      pc_(pc),
      predicted_next_pc_(predicted_next_pc),
      branch_prediction_(branch_prediction),
      fetch_sequence_(fetch_sequence) {}

const StaticInst& DynInst::staticInst() const noexcept {
    return *static_inst_;
}

std::uint64_t DynInst::pc() const noexcept {
    return pc_;
}

std::uint64_t DynInst::predictedNextPc() const noexcept {
    return predicted_next_pc_;
}

std::uint64_t DynInst::fetchSequence() const noexcept {
    return fetch_sequence_;
}

const BranchPrediction& DynInst::branchPrediction() const noexcept {
    return branch_prediction_;
}

RenameState& DynInst::renameState() noexcept {
    return rename_;
}

const RenameState& DynInst::renameState() const noexcept {
    return rename_;
}

ExecuteState& DynInst::executeState() noexcept {
    return execute_;
}

const ExecuteState& DynInst::executeState() const noexcept {
    return execute_;
}

CommitState& DynInst::commitState() noexcept {
    return commit_;
}

const CommitState& DynInst::commitState() const noexcept {
    return commit_;
}

DynInstPtr DynInstPool::create(const StaticInst& static_inst, std::uint64_t pc,
                               std::uint64_t predicted_next_pc,
                               const BranchPrediction& branch_prediction) {
    insts_.emplace_back(static_inst, pc, predicted_next_pc, branch_prediction,
                        next_fetch_sequence_++);
    return &insts_.back();
}

} // namespace riscv_cpu
