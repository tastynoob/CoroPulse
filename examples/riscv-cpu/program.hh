#pragma once

#include "isa.hh"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace riscv_cpu {

std::vector<StaticInst> buildProgram();
std::vector<StaticInst> buildSyntheticProgram(std::size_t instruction_count);
std::vector<std::uint8_t> loadRawImage(const std::string& path);
std::vector<StaticInst> loadRawProgram(const std::string& path);
std::uint64_t syntheticRegisterValue(std::size_t instruction_count, int reg);
void printProgram(const std::vector<StaticInst>& program, std::ostream& os,
                  std::size_t max_count);

} // namespace riscv_cpu
