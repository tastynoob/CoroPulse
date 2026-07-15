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
        publishRedirect();
        publishCompletion();
        completeReadyUop();
        publishRedirect();
        publishCompletion();

        if (auto issued = issue_in.read()) {
            auto* inst = *issued;
            inst->executeState().done_tick = currentTick() + inst->staticInst().latency;
            executing_.push_back(Executing{
                inst,
                inst->executeState().done_tick,
            });
            ++accepted_;
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
    if (pending_completion_ && completion_out.write(*pending_completion_)) {
        pending_completion_.reset();
        ++completed_;
    }
}

void ExecuteStage::publishRedirect() {
    if (pending_redirect_ && redirect_out.write(*pending_redirect_)) {
        pending_redirect_.reset();
    }
}

void ExecuteStage::completeReadyUop() {
    if (pending_completion_ || pending_redirect_) {
        return;
    }

    for (auto iter = executing_.begin(); iter != executing_.end(); ++iter) {
        if (iter->done_tick <= currentTick()) {
            pending_completion_ = execute(iter->inst);
            executing_.erase(iter);
            return;
        }
    }
}

ExecResult ExecuteStage::execute(DynInstPtr inst) {
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

    switch (static_inst.opcode) {
    case Opcode::lui:
        result.value = static_cast<std::uint64_t>(static_inst.imm);
        break;
    case Opcode::auipc:
        result.value = addSignedImmediate(inst->pc(), static_inst.imm);
        break;
    case Opcode::jal:
        result.value = inst->pc() + 4;
        pending_redirect_ = ControlRedirect{branchTarget(inst->pc(), static_inst.imm), false};
        break;
    case Opcode::jalr:
        result.value = inst->pc() + 4;
        pending_redirect_ =
            ControlRedirect{addSignedImmediate(src1, static_inst.imm) & ~std::uint64_t{1},
                            false};
        break;
    case Opcode::beq:
        pending_redirect_ = ControlRedirect{
            src1 == src2 ? branchTarget(inst->pc(), static_inst.imm) : inst->pc() + 4,
            false};
        break;
    case Opcode::bne:
        pending_redirect_ = ControlRedirect{
            src1 != src2 ? branchTarget(inst->pc(), static_inst.imm) : inst->pc() + 4,
            false};
        break;
    case Opcode::blt:
        pending_redirect_ = ControlRedirect{
            signed_src1 < signed_src2 ? branchTarget(inst->pc(), static_inst.imm)
                                      : inst->pc() + 4,
            false};
        break;
    case Opcode::bge:
        pending_redirect_ = ControlRedirect{
            signed_src1 >= signed_src2 ? branchTarget(inst->pc(), static_inst.imm)
                                       : inst->pc() + 4,
            false};
        break;
    case Opcode::bltu:
        pending_redirect_ = ControlRedirect{
            src1 < src2 ? branchTarget(inst->pc(), static_inst.imm) : inst->pc() + 4,
            false};
        break;
    case Opcode::bgeu:
        pending_redirect_ = ControlRedirect{
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
        sram_.store(addSignedImmediate(src1, static_inst.imm), src2, 1);
        break;
    case Opcode::sh:
        sram_.store(addSignedImmediate(src1, static_inst.imm), src2, 2);
        break;
    case Opcode::sw:
        sram_.store(addSignedImmediate(src1, static_inst.imm), src2, 4);
        break;
    case Opcode::sd:
        sram_.store(addSignedImmediate(src1, static_inst.imm), src2, 8);
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
        pending_redirect_ = ControlRedirect{inst->pc() + 4, true};
        break;
    case Opcode::illegal:
        break;
    }

    return result;
}

} // namespace riscv_cpu
