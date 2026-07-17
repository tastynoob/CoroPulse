#include "exec_context.hh"

#include "core_state.hh"
#include "inst.hh"
#include "memory.hh"

#include <utility>

namespace riscv_cpu {

ExecContext::ExecContext(CoreState& core, SimpleSram& sram, DynInst& inst,
                         coropulse::TickId tick)
    : core_(core), sram_(sram), inst_(inst) {
    sram_.setTimerValue(tick);
}

const StaticInst& ExecContext::staticInst() const noexcept {
    return inst_.staticInst();
}

std::uint64_t ExecContext::pc() const noexcept {
    return inst_.pc();
}

std::uint64_t ExecContext::predictedNextPc() const noexcept {
    return inst_.predictedNextPc();
}

const BranchPrediction& ExecContext::branchPrediction() const noexcept {
    return inst_.branchPrediction();
}

std::uint64_t ExecContext::readSrc1() const noexcept {
    return core_.readPhysicalRegister(inst_.renameState().src1.phys);
}

std::uint64_t ExecContext::readSrc2() const noexcept {
    return core_.readPhysicalRegister(inst_.renameState().src2.phys);
}

void ExecContext::writeResult(std::uint64_t value) noexcept {
    const auto& rename = inst_.renameState();
    if (rename.writes_rd) {
        core_.writePhysicalRegister(rename.phys_dst, value);
    }
}

std::uint64_t ExecContext::load(std::uint64_t address, std::size_t bytes,
                                bool sign_extend) const {
    return sram_.load(address, bytes, sign_extend);
}

void ExecContext::validateStore(std::uint64_t address,
                                std::size_t bytes) const {
    sram_.validateStore(address, bytes);
}

void ExecContext::setStore(std::uint64_t address, std::uint64_t value,
                           std::size_t bytes) {
    inst_.executeState().store = StoreWrite{address, value, bytes};
}

void ExecContext::setRedirect(ControlRedirect redirect) {
    inst_.executeState().redirect = redirect;
}

void ExecContext::setException(ExceptionCode code, std::uint64_t fault_address,
                               std::string message) {
    inst_.executeState().exception = ExceptionState{
        code,
        fault_address,
        std::move(message),
    };
}

void ExecContext::recordBranch(bool conditional, bool taken,
                               std::uint64_t target,
                               std::uint64_t actual_next_pc) {
    BranchUpdate update;
    update.valid = true;
    update.conditional = conditional;
    update.taken = taken;
    update.pc = inst_.pc();
    update.target_pc = target;
    update.actual_next_pc = actual_next_pc;
    update.predicted_next_pc = inst_.predictedNextPc();
    update.mispredicted = actual_next_pc != inst_.predictedNextPc();
    update.prediction = inst_.branchPrediction();

    auto& execute = inst_.executeState();
    execute.branch_update = update;
    execute.redirect = ControlRedirect{actual_next_pc, false};
}

void ExecContext::suppressPredictedRedirect() {
    auto& execute = inst_.executeState();
    if (execute.redirect && !execute.redirect->halt &&
        execute.redirect->next_pc == inst_.predictedNextPc()) {
        execute.redirect.reset();
    }
}

} // namespace riscv_cpu
