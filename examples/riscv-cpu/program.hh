#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace riscv_cpu {

std::vector<std::uint8_t> loadRawImage(const std::string& path);

} // namespace riscv_cpu
