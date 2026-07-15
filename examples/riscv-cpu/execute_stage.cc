#include "stages.hh"

#include <cstdint>

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

ExecuteStage::ExecuteStage(SimpleSram& sram) : sram_(sram) {}

coropulse::Task<void> ExecuteStage::process() {
    for (;; co_yield coropulse::tickDone{}) {
        if (redirect_in.read()) {
            executing_.clear();
            pending_completion_.clear();
            (void)issue_in.read();
            continue;
        }

        publishCompletion();
        completeReadyUops();
        publishCompletion();

        if (auto issued = issue_in.read()) {
            for (auto* inst : *issued) {
                inst->executeState().done_tick = currentTick() + inst->staticInst().latency;
                executing_.push_back(Executing{
                    inst,
                    inst->executeState().done_tick,
                });
            }
            accepted_ += issued->size();
        }
    }
    co_return;
}

std::size_t ExecuteStage::acceptedCount() const {
    return accepted_;
}

std::size_t ExecuteStage::completedCount() const {
    return completed_;
}

void ExecuteStage::publishCompletion() {
    if (!pending_completion_.empty() && completion_out.write(pending_completion_)) {
        completed_ += pending_completion_.size();
        pending_completion_.clear();
    }
}

void ExecuteStage::completeReadyUops() {
    if (!pending_completion_.empty()) {
        return;
    }

    for (auto iter = executing_.begin(); iter != executing_.end();) {
        if (iter->done_tick <= currentTick()) {
            pending_completion_.push_back(execute(iter->inst));
            executing_.erase(iter);
        } else {
            ++iter;
        }
    }
}

ExecResult ExecuteStage::execute(DynInstPtr inst) {
    sram_.setTimerValue(currentTick());

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

    switch (static_inst.opcode) {
    case Opcode::lui:
        result.value = static_cast<std::uint64_t>(static_inst.imm);
        break;
    case Opcode::auipc:
        result.value = addSignedImmediate(inst->pc(), static_inst.imm);
        break;
    case Opcode::jal:
        result.value = inst->pc() + 4;
        execute.redirect = ControlRedirect{branchTarget(inst->pc(), static_inst.imm), false};
        break;
    case Opcode::jalr:
        result.value = inst->pc() + 4;
        execute.redirect =
            ControlRedirect{addSignedImmediate(src1, static_inst.imm) & ~std::uint64_t{1},
                            false};
        break;
    case Opcode::beq:
        execute.redirect = ControlRedirect{
            src1 == src2 ? branchTarget(inst->pc(), static_inst.imm) : inst->pc() + 4,
            false};
        break;
    case Opcode::bne:
        execute.redirect = ControlRedirect{
            src1 != src2 ? branchTarget(inst->pc(), static_inst.imm) : inst->pc() + 4,
            false};
        break;
    case Opcode::blt:
        execute.redirect = ControlRedirect{
            signed_src1 < signed_src2 ? branchTarget(inst->pc(), static_inst.imm)
                                      : inst->pc() + 4,
            false};
        break;
    case Opcode::bge:
        execute.redirect = ControlRedirect{
            signed_src1 >= signed_src2 ? branchTarget(inst->pc(), static_inst.imm)
                                       : inst->pc() + 4,
            false};
        break;
    case Opcode::bltu:
        execute.redirect = ControlRedirect{
            src1 < src2 ? branchTarget(inst->pc(), static_inst.imm) : inst->pc() + 4,
            false};
        break;
    case Opcode::bgeu:
        execute.redirect = ControlRedirect{
            src1 >= src2 ? branchTarget(inst->pc(), static_inst.imm) : inst->pc() + 4,
            false};
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

    if (execute.redirect && !execute.redirect->halt &&
        execute.redirect->next_pc == inst->predictedNextPc()) {
        execute.redirect.reset();
    }

    return result;
}

} // namespace riscv_cpu
