#pragma once

#include "pipeline_types.hh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

namespace riscv_cpu {

class CoreState {
public:
    explicit CoreState(std::size_t physical_registers = 40);

    bool canRenameAny() const;
    void rename(DynInstPtr inst);
    void dispatchRenamed(DynInstPtr inst);
    void discardRenamed(DynInstPtr inst);
    void completeRedirectFlush();
    void markCompleted(const ExecResult& result);
    RetireResult retire(std::size_t max_count);
    std::size_t committedCount() const;
    std::size_t inFlightCount() const;
    std::uint64_t registerValue(int reg) const;
    std::size_t freePhysicalRegisters() const;
    std::size_t physicalRegisterCount() const noexcept;

private:
    struct RegisterState {
        std::uint64_t value = 0;
        std::size_t phys = 0;
        std::size_t committed_phys = 0;
        std::optional<std::size_t> producer;
    };

    struct RobEntry {
        DynInstPtr inst = nullptr;
    };

    Operand readSourceLocked(int reg) const;
    void flushAfterRedirectLocked(std::size_t redirect_sequence);
    void restoreSpeculativeRenameMapLocked();

    mutable std::mutex mutex_;
    std::array<RegisterState, 32> registers_{};
    std::vector<std::size_t> free_phys_regs_;
    std::size_t physical_register_count_ = 0;
    std::vector<RobEntry> rob_;
    std::size_t commit_head_ = 0;
    std::size_t committed_ = 0;
    std::size_t next_sequence_ = 0;
};

} // namespace riscv_cpu
