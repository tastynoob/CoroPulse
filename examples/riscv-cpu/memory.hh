#pragma once

#include "isa.hh"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

namespace riscv_cpu {

class SimpleSram {
public:
    explicit SimpleSram(std::vector<std::uint8_t> image, std::size_t data_bytes);

    std::size_t instructionCount() const noexcept;
    const StaticInst& loadInstruction(std::uint64_t pc) const;
    std::uint64_t load(std::uint64_t address, std::size_t bytes,
                       bool sign_extend) const;
    void store(std::uint64_t address, std::uint64_t value, std::size_t bytes);
    void validateLoad(std::uint64_t address, std::size_t bytes) const;
    void validateStore(std::uint64_t address, std::size_t bytes) const;
    std::uint64_t load64(std::uint64_t address) const;
    void store64(std::uint64_t address, std::uint64_t value);
    void setTimerValue(std::uint64_t value);

private:
    std::uint32_t loadInstructionWordLocked(std::size_t index) const;
    std::optional<std::uint64_t> loadMmioLocked(std::uint64_t address,
                                                std::size_t bytes) const;
    bool storeMmioLocked(std::uint64_t address, std::uint64_t value,
                         std::size_t bytes);
    void checkAccess(std::uint64_t address, std::size_t bytes) const;

    std::vector<std::uint8_t> instruction_bytes_;
    mutable std::vector<StaticInst> decoded_cache_;
    mutable std::vector<bool> decoded_valid_;
    mutable std::mutex mutex_;
    std::vector<std::uint8_t> data_;
    std::uint64_t timer_value_ = 0;
};

} // namespace riscv_cpu
