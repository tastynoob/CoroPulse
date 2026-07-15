#include "program.hh"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace riscv_cpu {
namespace {

std::uint32_t encodeR(std::uint32_t funct7, int rs2, int rs1, std::uint32_t funct3,
                      int rd, std::uint32_t opcode) {
    return (funct7 << 25) | (static_cast<std::uint32_t>(rs2) << 20) |
           (static_cast<std::uint32_t>(rs1) << 15) | (funct3 << 12) |
           (static_cast<std::uint32_t>(rd) << 7) | opcode;
}

std::uint32_t encodeI(std::int32_t imm, int rs1, std::uint32_t funct3, int rd,
                      std::uint32_t opcode) {
    return ((static_cast<std::uint32_t>(imm) & 0xfffU) << 20) |
           (static_cast<std::uint32_t>(rs1) << 15) | (funct3 << 12) |
           (static_cast<std::uint32_t>(rd) << 7) | opcode;
}

std::uint32_t encodeS(std::int32_t imm, int rs2, int rs1, std::uint32_t funct3,
                      std::uint32_t opcode) {
    const auto encoded = static_cast<std::uint32_t>(imm) & 0xfffU;
    return ((encoded >> 5) << 25) | (static_cast<std::uint32_t>(rs2) << 20) |
           (static_cast<std::uint32_t>(rs1) << 15) | (funct3 << 12) |
           ((encoded & 0x1fU) << 7) | opcode;
}

std::int64_t syntheticImmediate(std::size_t index) {
    return static_cast<std::int64_t>((index * 17 + 3) & 0x7ffULL);
}

std::vector<StaticInst> decodeWords(const std::vector<std::uint32_t>& words) {
    std::vector<StaticInst> program;
    program.reserve(words.size());
    for (auto word : words) {
        program.push_back(decodeInstruction(word));
    }
    return program;
}

} // namespace

std::vector<StaticInst> buildProgram() {
    return decodeWords({
        encodeI(5, 0, 0, 1, 0x13),      // addi x1, x0, 5
        encodeI(7, 0, 0, 2, 0x13),      // addi x2, x0, 7
        encodeR(0, 2, 1, 0, 3, 0x33),   // add x3, x1, x2
        encodeI(0, 0, 3, 4, 0x03),      // ld x4, 0(x0)
        encodeR(0, 4, 3, 0, 5, 0x33),   // add x5, x3, x4
        encodeI(1, 5, 0, 6, 0x13),      // addi x6, x5, 1
        encodeR(0, 2, 6, 0, 7, 0x33),   // add x7, x6, x2
        encodeS(8, 7, 0, 3, 0x23),      // sd x7, 8(x0)
    });
}

std::vector<StaticInst> buildSyntheticProgram(std::size_t instruction_count) {
    std::vector<std::uint32_t> words;
    words.reserve(instruction_count);

    for (std::size_t i = 0; i < instruction_count; ++i) {
        const int rd = static_cast<int>((i % 31) + 1);
        words.push_back(encodeI(static_cast<std::int32_t>(syntheticImmediate(i)),
                                0, 0, rd, 0x13));
    }
    return decodeWords(words);
}

std::vector<StaticInst> loadRawProgram(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open raw program: " + path);
    }

    std::vector<unsigned char> bytes(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    if (bytes.empty()) {
        throw std::runtime_error("raw program is empty: " + path);
    }
    if (bytes.size() % 4 != 0) {
        throw std::runtime_error("raw program size is not a multiple of 4 bytes");
    }

    std::vector<std::uint32_t> words;
    words.reserve(bytes.size() / 4);
    for (std::size_t i = 0; i < bytes.size(); i += 4) {
        words.push_back(static_cast<std::uint32_t>(bytes[i]) |
                        (static_cast<std::uint32_t>(bytes[i + 1]) << 8) |
                        (static_cast<std::uint32_t>(bytes[i + 2]) << 16) |
                        (static_cast<std::uint32_t>(bytes[i + 3]) << 24));
    }
    return decodeWords(words);
}

std::uint64_t syntheticRegisterValue(std::size_t instruction_count, int reg) {
    if (reg <= 0 || reg >= 32) {
        return 0;
    }

    const auto first_index = static_cast<std::size_t>(reg - 1);
    if (first_index >= instruction_count) {
        return 0;
    }

    const auto last_index =
        first_index + ((instruction_count - 1 - first_index) / 31) * 31;
    return static_cast<std::uint64_t>(syntheticImmediate(last_index));
}

void printProgram(const std::vector<StaticInst>& program, std::ostream& os,
                  std::size_t max_count) {
    os << "program:\n";
    const auto count = std::min(program.size(), max_count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& inst = program[i];
        os << "  pc=" << (i * 4)
           << " bits=0x" << std::hex << std::setw(8) << std::setfill('0')
           << inst.bits << std::dec << std::setfill(' ')
           << " " << opcodeName(inst.opcode)
           << " rd=x" << inst.rd
           << " rs1=x" << inst.rs1
           << " rs2=x" << inst.rs2
           << " imm=" << inst.imm << '\n';
    }
    if (count < program.size()) {
        os << "  ... " << (program.size() - count) << " more instructions\n";
    }
}

} // namespace riscv_cpu
