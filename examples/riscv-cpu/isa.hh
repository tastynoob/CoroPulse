#pragma once

#include <cstdint>

namespace riscv_cpu {

enum class Opcode {
    illegal,
    lui,
    auipc,
    jal,
    jalr,
    beq,
    bne,
    blt,
    bge,
    bltu,
    bgeu,
    lb,
    lh,
    lw,
    add,
    sub,
    sll,
    slt,
    sltu,
    xor_op,
    srl,
    sra,
    or_op,
    and_op,
    addi,
    slti,
    sltiu,
    xori,
    ori,
    andi,
    slli,
    srli,
    srai,
    lbu,
    lhu,
    lwu,
    ld,
    sb,
    sh,
    sw,
    sd,
    addiw,
    slliw,
    srliw,
    sraiw,
    addw,
    subw,
    sllw,
    srlw,
    sraw,
    fence,
    ecall,
    ebreak,
};

struct StaticInst {
    std::uint32_t bits = 0;
    Opcode opcode = Opcode::addi;
    int rd = 0;
    int rs1 = 0;
    int rs2 = 0;
    std::int64_t imm = 0;
    std::uint32_t latency = 1;
};

StaticInst decodeInstruction(std::uint32_t bits);
const char* opcodeName(Opcode opcode);
bool usesRs1(const StaticInst& inst);
bool usesRs2(const StaticInst& inst);
bool writesRd(const StaticInst& inst);
bool isMemory(const StaticInst& inst);
bool isControlFlow(const StaticInst& inst);
bool isHalt(const StaticInst& inst);

} // namespace riscv_cpu
