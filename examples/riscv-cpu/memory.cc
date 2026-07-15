#include "memory.hh"

#include <stdexcept>
#include <utility>

namespace riscv_cpu {

namespace {

std::uint64_t signExtend(std::uint64_t value, std::size_t bits) {
    const auto sign_bit = std::uint64_t{1} << (bits - 1);
    const auto mask = bits == 64 ? ~std::uint64_t{0} : ((std::uint64_t{1} << bits) - 1);
    value &= mask;
    return (value ^ sign_bit) - sign_bit;
}

} // namespace

SimpleSram::SimpleSram(std::vector<StaticInst> program, std::size_t data_bytes)
    : program_(std::move(program)),
      data_(data_bytes, 0) {}

std::size_t SimpleSram::instructionCount() const noexcept {
    return program_.size();
}

const StaticInst& SimpleSram::loadInstruction(std::uint64_t pc) const {
    if (pc % 4 != 0) {
        throw std::runtime_error("instruction fetch pc is not aligned");
    }
    const auto index = static_cast<std::size_t>(pc / 4);
    if (index >= program_.size()) {
        throw std::runtime_error("instruction fetch out of range");
    }
    return program_[index];
}

std::uint64_t SimpleSram::load64(std::uint64_t address) const {
    return load(address, 8, false);
}

void SimpleSram::store64(std::uint64_t address, std::uint64_t value) {
    store(address, value, 8);
}

std::uint64_t SimpleSram::load(std::uint64_t address, std::size_t bytes,
                               bool sign_extend) const {
    std::lock_guard lock(mutex_);
    checkAccess(address, bytes);

    std::uint64_t value = 0;
    for (std::size_t i = 0; i < bytes; ++i) {
        value |= static_cast<std::uint64_t>(data_[static_cast<std::size_t>(address) + i])
                 << (i * 8);
    }

    if (sign_extend && bytes < 8) {
        return signExtend(value, bytes * 8);
    }
    return value;
}

void SimpleSram::store(std::uint64_t address, std::uint64_t value, std::size_t bytes) {
    std::lock_guard lock(mutex_);
    checkAccess(address, bytes);

    for (std::size_t i = 0; i < bytes; ++i) {
        data_[static_cast<std::size_t>(address) + i] =
            static_cast<std::uint8_t>((value >> (i * 8)) & 0xffU);
    }
}

void SimpleSram::checkAccess(std::uint64_t address, std::size_t bytes) const {
    if (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8) {
        throw std::runtime_error("data sram access has unsupported width");
    }
    if (address % bytes != 0) {
        throw std::runtime_error("data sram access is not aligned");
    }
    if (address > data_.size() || bytes > data_.size() - static_cast<std::size_t>(address)) {
        throw std::runtime_error("data sram access out of range");
    }
}

} // namespace riscv_cpu
