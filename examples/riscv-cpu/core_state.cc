#include "core_state.hh"

#include "inst.hh"

#include <stdexcept>

namespace riscv_cpu {

CoreState::CoreState(std::size_t physical_registers)
    : physical_register_count_(physical_registers) {
    if (physical_registers < registers_.size()) {
        throw std::runtime_error("physical register count must cover architectural registers");
    }

    for (std::size_t i = 0; i < registers_.size(); ++i) {
        registers_[i].phys = i;
    }
    for (std::size_t phys = registers_.size(); phys < physical_registers; ++phys) {
        free_phys_regs_.push_back(phys);
    }
}

bool CoreState::canRenameAny() const {
    std::lock_guard lock(mutex_);
    return !free_phys_regs_.empty();
}

void CoreState::rename(DynInstPtr dyn_inst) {
    std::lock_guard lock(mutex_);
    if (!dyn_inst) {
        throw std::runtime_error("rename received a null dyninst");
    }

    const auto sequence = rob_.size();
    const auto& inst = dyn_inst->staticInst();
    const bool writes = writesRd(inst) && inst.rd != 0;

    if (writes && free_phys_regs_.empty()) {
        throw std::runtime_error("rename ran out of physical registers");
    }

    rob_.push_back(RobEntry{dyn_inst});

    auto& rename = dyn_inst->renameState();
    rename.sequence = sequence;
    rename.writes_rd = writes;
    rename.memory = isMemory(inst);

    if (usesRs1(inst)) {
        rename.src1 = readSourceLocked(inst.rs1);
    }
    if (usesRs2(inst)) {
        rename.src2 = readSourceLocked(inst.rs2);
    }

    if (writes) {
        auto& reg = registers_[static_cast<std::size_t>(inst.rd)];
        rename.old_phys_dst = reg.phys;
        rename.phys_dst = free_phys_regs_.back();
        free_phys_regs_.pop_back();

        reg.phys = rename.phys_dst;
        reg.producer = sequence;
    }
}

void CoreState::markCompleted(const ExecResult& result) {
    std::lock_guard lock(mutex_);
    if (result.sequence >= rob_.size()) {
        throw std::runtime_error("completion references an unknown rob entry");
    }
    auto& entry = rob_[result.sequence];
    if (entry.inst != result.inst) {
        throw std::runtime_error("completion dyninst does not match rob entry");
    }
    entry.inst->executeState().value = result.value;
    entry.inst->commitState().completed = true;
}

std::size_t CoreState::retire(std::size_t max_count) {
    std::lock_guard lock(mutex_);

    std::size_t retired = 0;
    while (retired < max_count && commit_head_ < rob_.size()) {
        const auto& entry = rob_[commit_head_];
        if (!entry.inst->commitState().completed) {
            break;
        }

        const auto& inst = entry.inst->staticInst();
        const auto& rename = entry.inst->renameState();
        if (rename.writes_rd) {
            auto& reg = registers_[static_cast<std::size_t>(inst.rd)];
            free_phys_regs_.push_back(rename.old_phys_dst);
            if (reg.producer && *reg.producer == rename.sequence) {
                reg.value = entry.inst->executeState().value;
                reg.producer.reset();
            }
        }

        ++commit_head_;
        ++committed_;
        ++retired;
    }
    return retired;
}

std::size_t CoreState::committedCount() const {
    std::lock_guard lock(mutex_);
    return committed_;
}

std::uint64_t CoreState::registerValue(int reg) const {
    std::lock_guard lock(mutex_);
    if (reg == 0) {
        return 0;
    }
    return registers_[static_cast<std::size_t>(reg)].value;
}

std::size_t CoreState::freePhysicalRegisters() const {
    std::lock_guard lock(mutex_);
    return free_phys_regs_.size();
}

std::size_t CoreState::physicalRegisterCount() const noexcept {
    return physical_register_count_;
}

Operand CoreState::readSourceLocked(int reg) const {
    if (reg == 0) {
        return Operand{true, 0, 0};
    }

    const auto& state = registers_[static_cast<std::size_t>(reg)];
    if (state.producer) {
        const auto* producer = rob_.at(*state.producer).inst;
        if (producer->commitState().completed) {
            return Operand{true, producer->executeState().value, 0};
        }
        return Operand{false, 0, *state.producer};
    }
    return Operand{true, state.value, 0};
}

} // namespace riscv_cpu
