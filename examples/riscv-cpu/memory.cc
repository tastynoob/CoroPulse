#include "memory.hh"

#include <algorithm>
#include <iostream>
#include <sstream>
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

constexpr std::uint64_t kUartBase = 0x10000000ULL;
constexpr std::uint64_t kUartData = kUartBase;
constexpr std::uint64_t kUartStatus = kUartBase + 8;
constexpr std::uint64_t kTimerBase = 0x20000000ULL;

std::uint64_t sliceBytes(std::uint64_t value, std::uint64_t address,
                         std::uint64_t base, std::size_t bytes) {
    const auto offset = static_cast<unsigned>((address - base) * 8);
    const auto shifted = value >> offset;
    if (bytes == 8) {
        return shifted;
    }
    const auto mask = (std::uint64_t{1} << (bytes * 8)) - 1;
    return shifted & mask;
}

} // namespace

SimpleSram::SimpleSram(std::vector<std::uint8_t> image, std::size_t data_bytes)
    : instruction_bytes_(std::move(image)),
      decoded_cache_(instruction_bytes_.size() / 4),
      decoded_valid_(instruction_bytes_.size() / 4, false),
      data_(std::max(data_bytes, instruction_bytes_.size()), 0) {
    if (instruction_bytes_.empty()) {
        throw std::runtime_error("raw image is empty");
    }
    if (instruction_bytes_.size() % 4 != 0) {
        throw std::runtime_error("raw image size is not a multiple of 4 bytes");
    }
    std::copy(instruction_bytes_.begin(), instruction_bytes_.end(), data_.begin());
}

std::size_t SimpleSram::instructionCount() const noexcept {
    return instruction_bytes_.size() / 4;
}

const StaticInst& SimpleSram::loadInstruction(std::uint64_t pc) const {
    if (pc % 4 != 0) {
        throw std::runtime_error("instruction fetch pc is not aligned");
    }
    const auto index = static_cast<std::size_t>(pc / 4);
    std::lock_guard lock(mutex_);
    if (index >= decoded_cache_.size()) {
        throw std::runtime_error("instruction fetch out of range");
    }
    if (!decoded_valid_[index]) {
        decoded_cache_[index] = decodeInstruction(loadInstructionWordLocked(index));
        decoded_valid_[index] = true;
    }
    return decoded_cache_[index];
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
    if (const auto mmio = loadMmioLocked(address, bytes)) {
        if (sign_extend && bytes < 8) {
            return signExtend(*mmio, bytes * 8);
        }
        return *mmio;
    }
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
    if (storeMmioLocked(address, value, bytes)) {
        return;
    }
    checkAccess(address, bytes);

    for (std::size_t i = 0; i < bytes; ++i) {
        data_[static_cast<std::size_t>(address) + i] =
            static_cast<std::uint8_t>((value >> (i * 8)) & 0xffU);
    }
}

void SimpleSram::validateLoad(std::uint64_t address, std::size_t bytes) const {
    std::lock_guard lock(mutex_);
    if (!loadMmioLocked(address, bytes)) {
        checkAccess(address, bytes);
    }
}

void SimpleSram::validateStore(std::uint64_t address, std::size_t bytes) const {
    std::lock_guard lock(mutex_);
    if (address == kUartData && bytes <= 8) {
        if (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8) {
            throw std::runtime_error("mmio store has unsupported width");
        }
        return;
    }
    if (address >= kTimerBase && address + bytes <= kTimerBase + 8) {
        if (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8) {
            throw std::runtime_error("mmio store has unsupported width");
        }
        if (address % bytes != 0) {
            throw std::runtime_error("mmio store is not aligned");
        }
        return;
    }
    checkAccess(address, bytes);
}

void SimpleSram::setTimerValue(std::uint64_t value) {
    std::lock_guard lock(mutex_);
    timer_value_ = value;
}

std::uint32_t SimpleSram::loadInstructionWordLocked(std::size_t index) const {
    const auto offset = index * 4;
    return static_cast<std::uint32_t>(instruction_bytes_[offset]) |
           (static_cast<std::uint32_t>(instruction_bytes_[offset + 1]) << 8) |
           (static_cast<std::uint32_t>(instruction_bytes_[offset + 2]) << 16) |
           (static_cast<std::uint32_t>(instruction_bytes_[offset + 3]) << 24);
}

std::optional<std::uint64_t> SimpleSram::loadMmioLocked(std::uint64_t address,
                                                        std::size_t bytes) const {
    if (address >= kTimerBase && address + bytes <= kTimerBase + 8) {
        if (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8) {
            throw std::runtime_error("mmio load has unsupported width");
        }
        if (address % bytes != 0) {
            throw std::runtime_error("mmio load is not aligned");
        }
        return sliceBytes(timer_value_, address, kTimerBase, bytes);
    }
    if (address == kUartStatus && bytes <= 8) {
        if (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8) {
            throw std::runtime_error("mmio load has unsupported width");
        }
        return 1;
    }
    if (address == kUartData && bytes <= 8) {
        if (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8) {
            throw std::runtime_error("mmio load has unsupported width");
        }
        return 0;
    }
    return std::nullopt;
}

bool SimpleSram::storeMmioLocked(std::uint64_t address, std::uint64_t value,
                                 std::size_t bytes) {
    if (address == kUartData && bytes <= 8) {
        if (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8) {
            throw std::runtime_error("mmio store has unsupported width");
        }
        const auto ch = static_cast<char>(value & 0xffU);
        std::cout.put(ch);
        if (ch == '\n') {
            std::cout.flush();
        }
        return true;
    }
    if (address >= kTimerBase && address + bytes <= kTimerBase + 8) {
        if (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8) {
            throw std::runtime_error("mmio store has unsupported width");
        }
        if (address % bytes != 0) {
            throw std::runtime_error("mmio store is not aligned");
        }
        return true;
    }
    return false;
}

void SimpleSram::checkAccess(std::uint64_t address, std::size_t bytes) const {
    if (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8) {
        throw std::runtime_error("data sram access has unsupported width");
    }
    if (address > data_.size() || bytes > data_.size() - static_cast<std::size_t>(address)) {
        std::ostringstream os;
        os << "data sram access out of range: address=0x"
           << std::hex << address << std::dec
           << ", bytes=" << bytes
           << ", data_size=" << data_.size();
        throw std::runtime_error(os.str());
    }
}

} // namespace riscv_cpu
