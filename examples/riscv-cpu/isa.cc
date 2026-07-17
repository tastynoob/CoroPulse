#include "isa.hh"

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace riscv_cpu {
namespace {

std::uint32_t slice(std::uint32_t value, unsigned low, unsigned width) {
    return (value >> low) & ((std::uint32_t{1} << width) - 1);
}

int reg(std::uint32_t bits, unsigned low) {
    return static_cast<int>(slice(bits, low, 5));
}

std::int64_t signExtend(std::uint64_t value, unsigned bits) {
    const auto sign_bit = std::uint64_t{1} << (bits - 1);
    const auto mask = (std::uint64_t{1} << bits) - 1;
    value &= mask;
    return static_cast<std::int64_t>((value ^ sign_bit) - sign_bit);
}

std::int64_t immI(std::uint32_t bits) {
    return signExtend(slice(bits, 20, 12), 12);
}

std::int64_t immS(std::uint32_t bits) {
    const auto imm = slice(bits, 7, 5) | (slice(bits, 25, 7) << 5);
    return signExtend(imm, 12);
}

std::int64_t immB(std::uint32_t bits) {
    const auto imm = (slice(bits, 8, 4) << 1) |
                     (slice(bits, 25, 6) << 5) |
                     (slice(bits, 7, 1) << 11) |
                     (slice(bits, 31, 1) << 12);
    return signExtend(imm, 13);
}

std::int64_t immU(std::uint32_t bits) {
    return signExtend(bits & 0xfffff000U, 32);
}

std::int64_t immJ(std::uint32_t bits) {
    const auto imm = (slice(bits, 21, 10) << 1) |
                     (slice(bits, 20, 1) << 11) |
                     (slice(bits, 12, 8) << 12) |
                     (slice(bits, 31, 1) << 20);
    return signExtend(imm, 21);
}

std::uint32_t shamt6(std::uint32_t bits) {
    return slice(bits, 20, 6);
}

std::uint32_t shamt5(std::uint32_t bits) {
    return slice(bits, 20, 5);
}

StaticInst makeInst(std::uint32_t bits, Opcode opcode, int rd = 0, int rs1 = 0,
                    int rs2 = 0, std::int64_t imm = 0) {
    return StaticInst{bits, opcode, rd, rs1, rs2, imm, 1};
}

[[noreturn]] void throwIllegal(std::uint32_t bits) {
    std::ostringstream os;
    os << "illegal or unsupported RV64I instruction 0x"
       << std::hex << std::setw(8) << std::setfill('0') << bits;
    throw std::runtime_error(os.str());
}

} // namespace

StaticInst decodeInstruction(std::uint32_t bits) {
    const auto opcode = slice(bits, 0, 7);
    const auto rd = reg(bits, 7);
    const auto funct3 = slice(bits, 12, 3);
    const auto rs1 = reg(bits, 15);
    const auto rs2 = reg(bits, 20);
    const auto funct7 = slice(bits, 25, 7);

    switch (opcode) {
    case 0x37:
        return makeInst(bits, Opcode::lui, rd, 0, 0, immU(bits));
    case 0x17:
        return makeInst(bits, Opcode::auipc, rd, 0, 0, immU(bits));
    case 0x6f:
        return makeInst(bits, Opcode::jal, rd, 0, 0, immJ(bits));
    case 0x67:
        if (funct3 == 0) {
            return makeInst(bits, Opcode::jalr, rd, rs1, 0, immI(bits));
        }
        break;
    case 0x63:
        switch (funct3) {
        case 0:
            return makeInst(bits, Opcode::beq, 0, rs1, rs2, immB(bits));
        case 1:
            return makeInst(bits, Opcode::bne, 0, rs1, rs2, immB(bits));
        case 4:
            return makeInst(bits, Opcode::blt, 0, rs1, rs2, immB(bits));
        case 5:
            return makeInst(bits, Opcode::bge, 0, rs1, rs2, immB(bits));
        case 6:
            return makeInst(bits, Opcode::bltu, 0, rs1, rs2, immB(bits));
        case 7:
            return makeInst(bits, Opcode::bgeu, 0, rs1, rs2, immB(bits));
        default:
            break;
        }
        break;
    case 0x03:
        switch (funct3) {
        case 0:
            return makeInst(bits, Opcode::lb, rd, rs1, 0, immI(bits));
        case 1:
            return makeInst(bits, Opcode::lh, rd, rs1, 0, immI(bits));
        case 2:
            return makeInst(bits, Opcode::lw, rd, rs1, 0, immI(bits));
        case 3:
            return makeInst(bits, Opcode::ld, rd, rs1, 0, immI(bits));
        case 4:
            return makeInst(bits, Opcode::lbu, rd, rs1, 0, immI(bits));
        case 5:
            return makeInst(bits, Opcode::lhu, rd, rs1, 0, immI(bits));
        case 6:
            return makeInst(bits, Opcode::lwu, rd, rs1, 0, immI(bits));
        default:
            break;
        }
        break;
    case 0x23:
        switch (funct3) {
        case 0:
            return makeInst(bits, Opcode::sb, 0, rs1, rs2, immS(bits));
        case 1:
            return makeInst(bits, Opcode::sh, 0, rs1, rs2, immS(bits));
        case 2:
            return makeInst(bits, Opcode::sw, 0, rs1, rs2, immS(bits));
        case 3:
            return makeInst(bits, Opcode::sd, 0, rs1, rs2, immS(bits));
        default:
            break;
        }
        break;
    case 0x13:
        switch (funct3) {
        case 0:
            return makeInst(bits, Opcode::addi, rd, rs1, 0, immI(bits));
        case 2:
            return makeInst(bits, Opcode::slti, rd, rs1, 0, immI(bits));
        case 3:
            return makeInst(bits, Opcode::sltiu, rd, rs1, 0, immI(bits));
        case 4:
            return makeInst(bits, Opcode::xori, rd, rs1, 0, immI(bits));
        case 6:
            return makeInst(bits, Opcode::ori, rd, rs1, 0, immI(bits));
        case 7:
            return makeInst(bits, Opcode::andi, rd, rs1, 0, immI(bits));
        case 1:
            if (slice(bits, 26, 6) == 0) {
                return makeInst(bits, Opcode::slli, rd, rs1, 0, shamt6(bits));
            }
            break;
        case 5:
            if (slice(bits, 26, 6) == 0) {
                return makeInst(bits, Opcode::srli, rd, rs1, 0, shamt6(bits));
            }
            if (slice(bits, 26, 6) == 0x10) {
                return makeInst(bits, Opcode::srai, rd, rs1, 0, shamt6(bits));
            }
            break;
        default:
            break;
        }
        break;
    case 0x33:
        switch (funct3) {
        case 0:
            if (funct7 == 0x00) {
                return makeInst(bits, Opcode::add, rd, rs1, rs2);
            }
            if (funct7 == 0x20) {
                return makeInst(bits, Opcode::sub, rd, rs1, rs2);
            }
            break;
        case 1:
            if (funct7 == 0x00) {
                return makeInst(bits, Opcode::sll, rd, rs1, rs2);
            }
            break;
        case 2:
            if (funct7 == 0x00) {
                return makeInst(bits, Opcode::slt, rd, rs1, rs2);
            }
            break;
        case 3:
            if (funct7 == 0x00) {
                return makeInst(bits, Opcode::sltu, rd, rs1, rs2);
            }
            break;
        case 4:
            if (funct7 == 0x00) {
                return makeInst(bits, Opcode::xor_op, rd, rs1, rs2);
            }
            break;
        case 5:
            if (funct7 == 0x00) {
                return makeInst(bits, Opcode::srl, rd, rs1, rs2);
            }
            if (funct7 == 0x20) {
                return makeInst(bits, Opcode::sra, rd, rs1, rs2);
            }
            break;
        case 6:
            if (funct7 == 0x00) {
                return makeInst(bits, Opcode::or_op, rd, rs1, rs2);
            }
            break;
        case 7:
            if (funct7 == 0x00) {
                return makeInst(bits, Opcode::and_op, rd, rs1, rs2);
            }
            break;
        default:
            break;
        }
        break;
    case 0x1b:
        switch (funct3) {
        case 0:
            return makeInst(bits, Opcode::addiw, rd, rs1, 0, immI(bits));
        case 1:
            if (funct7 == 0x00) {
                return makeInst(bits, Opcode::slliw, rd, rs1, 0, shamt5(bits));
            }
            break;
        case 5:
            if (funct7 == 0x00) {
                return makeInst(bits, Opcode::srliw, rd, rs1, 0, shamt5(bits));
            }
            if (funct7 == 0x20) {
                return makeInst(bits, Opcode::sraiw, rd, rs1, 0, shamt5(bits));
            }
            break;
        default:
            break;
        }
        break;
    case 0x3b:
        switch (funct3) {
        case 0:
            if (funct7 == 0x00) {
                return makeInst(bits, Opcode::addw, rd, rs1, rs2);
            }
            if (funct7 == 0x20) {
                return makeInst(bits, Opcode::subw, rd, rs1, rs2);
            }
            break;
        case 1:
            if (funct7 == 0x00) {
                return makeInst(bits, Opcode::sllw, rd, rs1, rs2);
            }
            break;
        case 5:
            if (funct7 == 0x00) {
                return makeInst(bits, Opcode::srlw, rd, rs1, rs2);
            }
            if (funct7 == 0x20) {
                return makeInst(bits, Opcode::sraw, rd, rs1, rs2);
            }
            break;
        default:
            break;
        }
        break;
    case 0x0f:
        return makeInst(bits, Opcode::fence);
    case 0x73:
        if (bits == 0x00000073) {
            return makeInst(bits, Opcode::ecall);
        }
        if (bits == 0x00100073) {
            return makeInst(bits, Opcode::ebreak);
        }
        break;
    default:
        break;
    }

    throwIllegal(bits);
}

const char* opcodeName(Opcode opcode) {
    switch (opcode) {
    case Opcode::illegal:
        return "illegal";
    case Opcode::lui:
        return "lui";
    case Opcode::auipc:
        return "auipc";
    case Opcode::jal:
        return "jal";
    case Opcode::jalr:
        return "jalr";
    case Opcode::beq:
        return "beq";
    case Opcode::bne:
        return "bne";
    case Opcode::blt:
        return "blt";
    case Opcode::bge:
        return "bge";
    case Opcode::bltu:
        return "bltu";
    case Opcode::bgeu:
        return "bgeu";
    case Opcode::lb:
        return "lb";
    case Opcode::lh:
        return "lh";
    case Opcode::lw:
        return "lw";
    case Opcode::ld:
        return "ld";
    case Opcode::lbu:
        return "lbu";
    case Opcode::lhu:
        return "lhu";
    case Opcode::lwu:
        return "lwu";
    case Opcode::sb:
        return "sb";
    case Opcode::sh:
        return "sh";
    case Opcode::sw:
        return "sw";
    case Opcode::sd:
        return "sd";
    case Opcode::addi:
        return "addi";
    case Opcode::slti:
        return "slti";
    case Opcode::sltiu:
        return "sltiu";
    case Opcode::xori:
        return "xori";
    case Opcode::ori:
        return "ori";
    case Opcode::andi:
        return "andi";
    case Opcode::slli:
        return "slli";
    case Opcode::srli:
        return "srli";
    case Opcode::srai:
        return "srai";
    case Opcode::add:
        return "add";
    case Opcode::sub:
        return "sub";
    case Opcode::sll:
        return "sll";
    case Opcode::slt:
        return "slt";
    case Opcode::sltu:
        return "sltu";
    case Opcode::xor_op:
        return "xor";
    case Opcode::srl:
        return "srl";
    case Opcode::sra:
        return "sra";
    case Opcode::or_op:
        return "or";
    case Opcode::and_op:
        return "and";
    case Opcode::addiw:
        return "addiw";
    case Opcode::slliw:
        return "slliw";
    case Opcode::srliw:
        return "srliw";
    case Opcode::sraiw:
        return "sraiw";
    case Opcode::addw:
        return "addw";
    case Opcode::subw:
        return "subw";
    case Opcode::sllw:
        return "sllw";
    case Opcode::srlw:
        return "srlw";
    case Opcode::sraw:
        return "sraw";
    case Opcode::fence:
        return "fence";
    case Opcode::ecall:
        return "ecall";
    case Opcode::ebreak:
        return "ebreak";
    }
    return "unknown";
}

bool usesRs1(const StaticInst& inst) {
    switch (inst.opcode) {
    case Opcode::jalr:
    case Opcode::beq:
    case Opcode::bne:
    case Opcode::blt:
    case Opcode::bge:
    case Opcode::bltu:
    case Opcode::bgeu:
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
    case Opcode::addi:
    case Opcode::slti:
    case Opcode::sltiu:
    case Opcode::xori:
    case Opcode::ori:
    case Opcode::andi:
    case Opcode::slli:
    case Opcode::srli:
    case Opcode::srai:
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
    case Opcode::addiw:
    case Opcode::slliw:
    case Opcode::srliw:
    case Opcode::sraiw:
    case Opcode::addw:
    case Opcode::subw:
    case Opcode::sllw:
    case Opcode::srlw:
    case Opcode::sraw:
        return true;
    default:
        return false;
    }
}

bool usesRs2(const StaticInst& inst) {
    switch (inst.opcode) {
    case Opcode::beq:
    case Opcode::bne:
    case Opcode::blt:
    case Opcode::bge:
    case Opcode::bltu:
    case Opcode::bgeu:
    case Opcode::sb:
    case Opcode::sh:
    case Opcode::sw:
    case Opcode::sd:
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
    case Opcode::addw:
    case Opcode::subw:
    case Opcode::sllw:
    case Opcode::srlw:
    case Opcode::sraw:
        return true;
    default:
        return false;
    }
}

bool writesRd(const StaticInst& inst) {
    switch (inst.opcode) {
    case Opcode::lui:
    case Opcode::auipc:
    case Opcode::jal:
    case Opcode::jalr:
    case Opcode::lb:
    case Opcode::lh:
    case Opcode::lw:
    case Opcode::ld:
    case Opcode::lbu:
    case Opcode::lhu:
    case Opcode::lwu:
    case Opcode::addi:
    case Opcode::slti:
    case Opcode::sltiu:
    case Opcode::xori:
    case Opcode::ori:
    case Opcode::andi:
    case Opcode::slli:
    case Opcode::srli:
    case Opcode::srai:
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
    case Opcode::addiw:
    case Opcode::slliw:
    case Opcode::srliw:
    case Opcode::sraiw:
    case Opcode::addw:
    case Opcode::subw:
    case Opcode::sllw:
    case Opcode::srlw:
    case Opcode::sraw:
        return true;
    default:
        return false;
    }
}

bool isLoad(const StaticInst& inst) {
    switch (inst.opcode) {
    case Opcode::lb:
    case Opcode::lh:
    case Opcode::lw:
    case Opcode::ld:
    case Opcode::lbu:
    case Opcode::lhu:
    case Opcode::lwu:
        return true;
    default:
        return false;
    }
}

bool isStore(const StaticInst& inst) {
    switch (inst.opcode) {
    case Opcode::sb:
    case Opcode::sh:
    case Opcode::sw:
    case Opcode::sd:
        return true;
    default:
        return false;
    }
}

bool isMemory(const StaticInst& inst) {
    return isLoad(inst) || isStore(inst);
}

std::size_t memoryAccessBytes(const StaticInst& inst) {
    switch (inst.opcode) {
    case Opcode::lb:
    case Opcode::lbu:
    case Opcode::sb:
        return 1;
    case Opcode::lh:
    case Opcode::lhu:
    case Opcode::sh:
        return 2;
    case Opcode::lw:
    case Opcode::lwu:
    case Opcode::sw:
        return 4;
    case Opcode::ld:
    case Opcode::sd:
        return 8;
    default:
        return 0;
    }
}

bool loadSignExtends(const StaticInst& inst) {
    switch (inst.opcode) {
    case Opcode::lb:
    case Opcode::lh:
    case Opcode::lw:
        return true;
    default:
        return false;
    }
}

bool isControlFlow(const StaticInst& inst) {
    switch (inst.opcode) {
    case Opcode::jal:
    case Opcode::jalr:
    case Opcode::beq:
    case Opcode::bne:
    case Opcode::blt:
    case Opcode::bge:
    case Opcode::bltu:
    case Opcode::bgeu:
    case Opcode::ecall:
    case Opcode::ebreak:
        return true;
    default:
        return false;
    }
}

bool isHalt(const StaticInst& inst) {
    return inst.opcode == Opcode::ecall || inst.opcode == Opcode::ebreak;
}

} // namespace riscv_cpu
