#include "backend_pipeline.hh"

#include <cstdint>
#include <optional>

namespace riscv_cpu {
namespace {

std::uint64_t signExtend32(std::uint64_t value) {
    const auto word = static_cast<std::uint32_t>(value);
    return static_cast<std::uint64_t>(static_cast<std::int64_t>(
        static_cast<std::int32_t>(word)));
}

std::uint64_t addSignedImmediate(std::uint64_t value, std::int64_t imm) {
    return static_cast<std::uint64_t>(static_cast<std::int64_t>(value) + imm);
}

std::uint64_t branchTarget(std::uint64_t pc, std::int64_t imm) {
    return static_cast<std::uint64_t>(static_cast<std::int64_t>(pc) + imm);
}

} // namespace

ExecutePipe::ExecutePipe(SimpleSram& sram) : sram_(sram) {}

void ExecutePipe::accept(InstBundle&& bundle, coropulse::TickId tick,
                         BackendStats& stats) {
    for (auto* inst : bundle) {
        inst->executeState().done_tick = tick + inst->staticInst().latency;
        executing_.push_back(Executing{
            inst,
            inst->executeState().done_tick,
        });
    }
    stats.execute_accepted += bundle.size();
}

ExecResultBundle ExecutePipe::collectCompletions(coropulse::TickId tick,
                                                 BackendStats& stats) {
    ExecResultBundle completed;
    if (!pending_completion_.empty()) {
        completed = std::move(pending_completion_);
        pending_completion_.clear();
        stats.execute_completed += completed.size();
    }

    completeReadyUops(tick);
    return completed;
}

void ExecutePipe::clear() {
    executing_.clear();
    pending_completion_.clear();
}

void ExecutePipe::completeReadyUops(coropulse::TickId tick) {
    if (!pending_completion_.empty()) {
        return;
    }

    for (auto iter = executing_.begin(); iter != executing_.end();) {
        if (iter->done_tick <= tick) {
            pending_completion_.push_back(execute(iter->inst, tick));
            executing_.erase(iter);
        } else {
            ++iter;
        }
    }
}

ExecResult ExecutePipe::execute(DynInstPtr inst, coropulse::TickId tick) {
    sram_.setTimerValue(tick);

    const auto& static_inst = inst->staticInst();
    const auto& rename = inst->renameState();
    const auto src1 = rename.src1.value;
    const auto src2 = rename.src2.value;
    const auto signed_src1 = static_cast<std::int64_t>(src1);
    const auto signed_src2 = static_cast<std::int64_t>(src2);

    ExecResult result;
    result.sequence = rename.sequence;
    result.inst = inst;
    result.writes_rd = rename.writes_rd;
    auto& execute = inst->executeState();

    std::optional<BranchUpdate> branch_update;
    const auto recordBranch = [&](bool conditional, bool taken,
                                  std::uint64_t target,
                                  std::uint64_t actual_next_pc) {
        BranchUpdate update;
        update.valid = true;
        update.conditional = conditional;
        update.taken = taken;
        update.pc = inst->pc();
        update.target_pc = target;
        update.actual_next_pc = actual_next_pc;
        update.predicted_next_pc = inst->predictedNextPc();
        update.mispredicted = actual_next_pc != inst->predictedNextPc();
        update.prediction = inst->branchPrediction();
        branch_update = update;
    };

    switch (static_inst.opcode) {
    case Opcode::lui:
        result.value = static_cast<std::uint64_t>(static_inst.imm);
        break;
    case Opcode::auipc:
        result.value = addSignedImmediate(inst->pc(), static_inst.imm);
        break;
    case Opcode::jal:
        result.value = inst->pc() + 4;
        recordBranch(false, true, branchTarget(inst->pc(), static_inst.imm),
                     branchTarget(inst->pc(), static_inst.imm));
        execute.redirect = ControlRedirect{branch_update->actual_next_pc, false};
        break;
    case Opcode::jalr:
        result.value = inst->pc() + 4;
        recordBranch(false, true,
                     addSignedImmediate(src1, static_inst.imm) & ~std::uint64_t{1},
                     addSignedImmediate(src1, static_inst.imm) & ~std::uint64_t{1});
        execute.redirect = ControlRedirect{branch_update->actual_next_pc, false};
        break;
    case Opcode::beq:
        recordBranch(true, src1 == src2, branchTarget(inst->pc(), static_inst.imm),
                     src1 == src2 ? branchTarget(inst->pc(), static_inst.imm)
                                  : inst->pc() + 4);
        execute.redirect = ControlRedirect{branch_update->actual_next_pc, false};
        break;
    case Opcode::bne:
        recordBranch(true, src1 != src2, branchTarget(inst->pc(), static_inst.imm),
                     src1 != src2 ? branchTarget(inst->pc(), static_inst.imm)
                                  : inst->pc() + 4);
        execute.redirect = ControlRedirect{branch_update->actual_next_pc, false};
        break;
    case Opcode::blt:
        recordBranch(true, signed_src1 < signed_src2,
                     branchTarget(inst->pc(), static_inst.imm),
                     signed_src1 < signed_src2
                         ? branchTarget(inst->pc(), static_inst.imm)
                         : inst->pc() + 4);
        execute.redirect = ControlRedirect{branch_update->actual_next_pc, false};
        break;
    case Opcode::bge:
        recordBranch(true, signed_src1 >= signed_src2,
                     branchTarget(inst->pc(), static_inst.imm),
                     signed_src1 >= signed_src2
                         ? branchTarget(inst->pc(), static_inst.imm)
                         : inst->pc() + 4);
        execute.redirect = ControlRedirect{branch_update->actual_next_pc, false};
        break;
    case Opcode::bltu:
        recordBranch(true, src1 < src2, branchTarget(inst->pc(), static_inst.imm),
                     src1 < src2 ? branchTarget(inst->pc(), static_inst.imm)
                                 : inst->pc() + 4);
        execute.redirect = ControlRedirect{branch_update->actual_next_pc, false};
        break;
    case Opcode::bgeu:
        recordBranch(true, src1 >= src2, branchTarget(inst->pc(), static_inst.imm),
                     src1 >= src2 ? branchTarget(inst->pc(), static_inst.imm)
                                  : inst->pc() + 4);
        execute.redirect = ControlRedirect{branch_update->actual_next_pc, false};
        break;
    case Opcode::lb:
        result.value = sram_.load(addSignedImmediate(src1, static_inst.imm), 1, true);
        break;
    case Opcode::lh:
        result.value = sram_.load(addSignedImmediate(src1, static_inst.imm), 2, true);
        break;
    case Opcode::lw:
        result.value = sram_.load(addSignedImmediate(src1, static_inst.imm), 4, true);
        break;
    case Opcode::ld:
        result.value = sram_.load(addSignedImmediate(src1, static_inst.imm), 8, false);
        break;
    case Opcode::lbu:
        result.value = sram_.load(addSignedImmediate(src1, static_inst.imm), 1, false);
        break;
    case Opcode::lhu:
        result.value = sram_.load(addSignedImmediate(src1, static_inst.imm), 2, false);
        break;
    case Opcode::lwu:
        result.value = sram_.load(addSignedImmediate(src1, static_inst.imm), 4, false);
        break;
    case Opcode::sb:
        result.store = StoreWrite{addSignedImmediate(src1, static_inst.imm), src2, 1};
        break;
    case Opcode::sh:
        result.store = StoreWrite{addSignedImmediate(src1, static_inst.imm), src2, 2};
        break;
    case Opcode::sw:
        result.store = StoreWrite{addSignedImmediate(src1, static_inst.imm), src2, 4};
        break;
    case Opcode::sd:
        result.store = StoreWrite{addSignedImmediate(src1, static_inst.imm), src2, 8};
        break;
    case Opcode::addi:
        result.value = addSignedImmediate(src1, static_inst.imm);
        break;
    case Opcode::slti:
        result.value = signed_src1 < static_inst.imm ? 1 : 0;
        break;
    case Opcode::sltiu:
        result.value = src1 < static_cast<std::uint64_t>(static_inst.imm) ? 1 : 0;
        break;
    case Opcode::xori:
        result.value = src1 ^ static_cast<std::uint64_t>(static_inst.imm);
        break;
    case Opcode::ori:
        result.value = src1 | static_cast<std::uint64_t>(static_inst.imm);
        break;
    case Opcode::andi:
        result.value = src1 & static_cast<std::uint64_t>(static_inst.imm);
        break;
    case Opcode::slli:
        result.value = src1 << (static_cast<unsigned>(static_inst.imm) & 0x3fU);
        break;
    case Opcode::srli:
        result.value = src1 >> (static_cast<unsigned>(static_inst.imm) & 0x3fU);
        break;
    case Opcode::srai:
        result.value = static_cast<std::uint64_t>(
            signed_src1 >> (static_cast<unsigned>(static_inst.imm) & 0x3fU));
        break;
    case Opcode::add:
        result.value = src1 + src2;
        break;
    case Opcode::sub:
        result.value = src1 - src2;
        break;
    case Opcode::sll:
        result.value = src1 << (src2 & 0x3fU);
        break;
    case Opcode::slt:
        result.value = signed_src1 < signed_src2 ? 1 : 0;
        break;
    case Opcode::sltu:
        result.value = src1 < src2 ? 1 : 0;
        break;
    case Opcode::xor_op:
        result.value = src1 ^ src2;
        break;
    case Opcode::srl:
        result.value = src1 >> (src2 & 0x3fU);
        break;
    case Opcode::sra:
        result.value = static_cast<std::uint64_t>(signed_src1 >> (src2 & 0x3fU));
        break;
    case Opcode::or_op:
        result.value = src1 | src2;
        break;
    case Opcode::and_op:
        result.value = src1 & src2;
        break;
    case Opcode::addiw:
        result.value = signExtend32(addSignedImmediate(src1, static_inst.imm));
        break;
    case Opcode::slliw:
        result.value = signExtend32(static_cast<std::uint32_t>(src1)
                                    << (static_cast<unsigned>(static_inst.imm) & 0x1fU));
        break;
    case Opcode::srliw:
        result.value = signExtend32(static_cast<std::uint32_t>(src1)
                                    >> (static_cast<unsigned>(static_inst.imm) & 0x1fU));
        break;
    case Opcode::sraiw:
        result.value = signExtend32(static_cast<std::uint32_t>(
            static_cast<std::int32_t>(src1) >>
            (static_cast<unsigned>(static_inst.imm) & 0x1fU)));
        break;
    case Opcode::addw:
        result.value = signExtend32(static_cast<std::uint32_t>(src1 + src2));
        break;
    case Opcode::subw:
        result.value = signExtend32(static_cast<std::uint32_t>(src1 - src2));
        break;
    case Opcode::sllw:
        result.value = signExtend32(static_cast<std::uint32_t>(src1)
                                    << (src2 & 0x1fU));
        break;
    case Opcode::srlw:
        result.value = signExtend32(static_cast<std::uint32_t>(src1)
                                    >> (src2 & 0x1fU));
        break;
    case Opcode::sraw:
        result.value = signExtend32(static_cast<std::uint32_t>(
            static_cast<std::int32_t>(src1) >> (src2 & 0x1fU)));
        break;
    case Opcode::fence:
        break;
    case Opcode::ecall:
    case Opcode::ebreak:
        execute.redirect = ControlRedirect{inst->pc() + 4, true};
        break;
    case Opcode::illegal:
        break;
    }

    if (branch_update) {
        execute.branch_update = branch_update;
    }

    if (execute.redirect && !execute.redirect->halt &&
        execute.redirect->next_pc == inst->predictedNextPc()) {
        execute.redirect.reset();
    }

    return result;
}

} // namespace riscv_cpu
