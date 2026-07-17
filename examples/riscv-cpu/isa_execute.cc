#include "isa_execute.hh"

#include "exec_context.hh"

#include <cstdint>
#include <stdexcept>

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

void executeJump(ExecContext& context, std::uint64_t src1) {
    const auto& static_inst = context.staticInst();

    switch (static_inst.opcode) {
    case Opcode::jal:
        context.writeResult(context.pc() + 4);
        context.recordBranch(false, true,
                             branchTarget(context.pc(), static_inst.imm),
                             branchTarget(context.pc(), static_inst.imm));
        break;
    case Opcode::jalr: {
        context.writeResult(context.pc() + 4);
        const auto target =
            addSignedImmediate(src1, static_inst.imm) & ~std::uint64_t{1};
        context.recordBranch(false, true, target, target);
        break;
    }
    default:
        break;
    }
}

void executeBranch(ExecContext& context, std::uint64_t src1,
                   std::uint64_t src2) {
    const auto& static_inst = context.staticInst();
    const auto signed_src1 = static_cast<std::int64_t>(src1);
    const auto signed_src2 = static_cast<std::int64_t>(src2);
    bool taken = false;

    switch (static_inst.opcode) {
    case Opcode::beq:
        taken = src1 == src2;
        break;
    case Opcode::bne:
        taken = src1 != src2;
        break;
    case Opcode::blt:
        taken = signed_src1 < signed_src2;
        break;
    case Opcode::bge:
        taken = signed_src1 >= signed_src2;
        break;
    case Opcode::bltu:
        taken = src1 < src2;
        break;
    case Opcode::bgeu:
        taken = src1 >= src2;
        break;
    default:
        return;
    }

    const auto target = branchTarget(context.pc(), static_inst.imm);
    context.recordBranch(true, taken, target,
                         taken ? target : context.pc() + 4);
}

void executeIntegerImmediate(ExecContext& context, std::uint64_t src1) {
    const auto& static_inst = context.staticInst();
    const auto signed_src1 = static_cast<std::int64_t>(src1);

    switch (static_inst.opcode) {
    case Opcode::lui:
        context.writeResult(static_cast<std::uint64_t>(static_inst.imm));
        break;
    case Opcode::auipc:
        context.writeResult(addSignedImmediate(context.pc(), static_inst.imm));
        break;
    case Opcode::addi:
        context.writeResult(addSignedImmediate(src1, static_inst.imm));
        break;
    case Opcode::slti:
        context.writeResult(signed_src1 < static_inst.imm ? 1 : 0);
        break;
    case Opcode::sltiu:
        context.writeResult(src1 < static_cast<std::uint64_t>(static_inst.imm)
                                ? 1
                                : 0);
        break;
    case Opcode::xori:
        context.writeResult(src1 ^ static_cast<std::uint64_t>(static_inst.imm));
        break;
    case Opcode::ori:
        context.writeResult(src1 | static_cast<std::uint64_t>(static_inst.imm));
        break;
    case Opcode::andi:
        context.writeResult(src1 & static_cast<std::uint64_t>(static_inst.imm));
        break;
    case Opcode::slli:
        context.writeResult(src1 << (static_cast<unsigned>(static_inst.imm) & 0x3fU));
        break;
    case Opcode::srli:
        context.writeResult(src1 >> (static_cast<unsigned>(static_inst.imm) & 0x3fU));
        break;
    case Opcode::srai:
        context.writeResult(static_cast<std::uint64_t>(
            signed_src1 >> (static_cast<unsigned>(static_inst.imm) & 0x3fU)));
        break;
    default:
        break;
    }
}

void executeIntegerRegister(ExecContext& context, std::uint64_t src1,
                            std::uint64_t src2) {
    const auto& static_inst = context.staticInst();
    const auto signed_src1 = static_cast<std::int64_t>(src1);
    const auto signed_src2 = static_cast<std::int64_t>(src2);

    switch (static_inst.opcode) {
    case Opcode::add:
        context.writeResult(src1 + src2);
        break;
    case Opcode::sub:
        context.writeResult(src1 - src2);
        break;
    case Opcode::sll:
        context.writeResult(src1 << (src2 & 0x3fU));
        break;
    case Opcode::slt:
        context.writeResult(signed_src1 < signed_src2 ? 1 : 0);
        break;
    case Opcode::sltu:
        context.writeResult(src1 < src2 ? 1 : 0);
        break;
    case Opcode::xor_op:
        context.writeResult(src1 ^ src2);
        break;
    case Opcode::srl:
        context.writeResult(src1 >> (src2 & 0x3fU));
        break;
    case Opcode::sra:
        context.writeResult(static_cast<std::uint64_t>(signed_src1 >> (src2 & 0x3fU)));
        break;
    case Opcode::or_op:
        context.writeResult(src1 | src2);
        break;
    case Opcode::and_op:
        context.writeResult(src1 & src2);
        break;
    default:
        break;
    }
}

void executeWordImmediate(ExecContext& context, std::uint64_t src1) {
    const auto& static_inst = context.staticInst();

    switch (static_inst.opcode) {
    case Opcode::addiw:
        context.writeResult(signExtend32(addSignedImmediate(src1, static_inst.imm)));
        break;
    case Opcode::slliw:
        context.writeResult(signExtend32(
            static_cast<std::uint32_t>(src1)
            << (static_cast<unsigned>(static_inst.imm) & 0x1fU)));
        break;
    case Opcode::srliw:
        context.writeResult(signExtend32(
            static_cast<std::uint32_t>(src1)
            >> (static_cast<unsigned>(static_inst.imm) & 0x1fU)));
        break;
    case Opcode::sraiw:
        context.writeResult(signExtend32(static_cast<std::uint32_t>(
            static_cast<std::int32_t>(src1) >>
            (static_cast<unsigned>(static_inst.imm) & 0x1fU))));
        break;
    default:
        break;
    }
}

void executeWordRegister(ExecContext& context, std::uint64_t src1,
                         std::uint64_t src2) {
    const auto& static_inst = context.staticInst();

    switch (static_inst.opcode) {
    case Opcode::addw:
        context.writeResult(signExtend32(static_cast<std::uint32_t>(src1 + src2)));
        break;
    case Opcode::subw:
        context.writeResult(signExtend32(static_cast<std::uint32_t>(src1 - src2)));
        break;
    case Opcode::sllw:
        context.writeResult(signExtend32(static_cast<std::uint32_t>(src1)
                                         << (src2 & 0x1fU)));
        break;
    case Opcode::srlw:
        context.writeResult(signExtend32(static_cast<std::uint32_t>(src1)
                                         >> (src2 & 0x1fU)));
        break;
    case Opcode::sraw:
        context.writeResult(signExtend32(static_cast<std::uint32_t>(
            static_cast<std::int32_t>(src1) >> (src2 & 0x1fU))));
        break;
    default:
        break;
    }
}

} // namespace

void executeInst(ExecContext& context) {
    const auto& static_inst = context.staticInst();
    const auto src1 = context.readSrc1();
    const auto src2 = context.readSrc2();

    switch (static_inst.opcode) {
    case Opcode::jal:
    case Opcode::jalr:
        executeJump(context, src1);
        break;
    case Opcode::beq:
    case Opcode::bne:
    case Opcode::blt:
    case Opcode::bge:
    case Opcode::bltu:
    case Opcode::bgeu:
        executeBranch(context, src1, src2);
        break;
    case Opcode::lb:
    case Opcode::lh:
    case Opcode::lw:
    case Opcode::ld:
    case Opcode::lbu:
    case Opcode::lhu:
    case Opcode::lwu:
    case Opcode::sb:
    case Opcode::sh:
    case Opcode::sw:
    case Opcode::sd:
        throw std::runtime_error("memory opcode entered integer execute");
    case Opcode::lui:
    case Opcode::auipc:
    case Opcode::addi:
    case Opcode::slti:
    case Opcode::sltiu:
    case Opcode::xori:
    case Opcode::ori:
    case Opcode::andi:
    case Opcode::slli:
    case Opcode::srli:
    case Opcode::srai:
        executeIntegerImmediate(context, src1);
        break;
    case Opcode::add:
    case Opcode::sub:
    case Opcode::sll:
    case Opcode::slt:
    case Opcode::sltu:
    case Opcode::xor_op:
    case Opcode::srl:
    case Opcode::sra:
    case Opcode::or_op:
    case Opcode::and_op:
        executeIntegerRegister(context, src1, src2);
        break;
    case Opcode::addiw:
    case Opcode::slliw:
    case Opcode::srliw:
    case Opcode::sraiw:
        executeWordImmediate(context, src1);
        break;
    case Opcode::addw:
    case Opcode::subw:
    case Opcode::sllw:
    case Opcode::srlw:
    case Opcode::sraw:
        executeWordRegister(context, src1, src2);
        break;
    case Opcode::fence:
        break;
    case Opcode::ecall:
    case Opcode::ebreak:
        context.setRedirect(ControlRedirect{context.pc() + 4, true});
        break;
    case Opcode::illegal:
        break;
    }

    context.suppressPredictedRedirect();
}

} // namespace riscv_cpu
