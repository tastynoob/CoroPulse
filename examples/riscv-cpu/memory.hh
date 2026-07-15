#pragma once

#include "isa.hh"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace riscv_cpu {

class SimpleSram {
public:
    explicit SimpleSram(std::vector<StaticInst> program, std::size_t data_bytes);

    std::size_t instructionCount() const noexcept;
    const StaticInst& loadInstruction(std::uint64_t pc) const;
    std::uint64_t load(std::uint64_t address, std::size_t bytes,
                       bool sign_extend) const;
    void store(std::uint64_t address, std::uint64_t value, std::size_t bytes);
    std::uint64_t load64(std::uint64_t address) const;
    void store64(std::uint64_t address, std::uint64_t value);

private:
    void checkAccess(std::uint64_t address, std::size_t bytes) const;

    std::vector<StaticInst> program_;
    mutable std::mutex mutex_;
    std::vector<std::uint8_t> data_;
};

} // namespace riscv_cpu
