#include "core_state.hh"

#include "inst.hh"

#include <sstream>
#include <stdexcept>

namespace riscv_cpu {

CoreState::CoreState(std::size_t physical_registers, std::size_t rob_capacity)
    : physical_register_count_(physical_registers),
      rob_capacity_(rob_capacity),
      physical_regs_(physical_registers, 0) {
    if (physical_registers < registers_.size()) {
        throw std::runtime_error("physical register count must cover architectural registers");
    }
    if (rob_capacity_ == 0) {
        throw std::runtime_error("rob capacity must be greater than zero");
    }

    for (std::size_t i = 0; i < registers_.size(); ++i) {
        registers_[i].phys = i;
        registers_[i].committed_phys = i;
    }
    for (std::size_t phys = registers_.size(); phys < physical_registers; ++phys) {
        free_phys_regs_.push_back(phys);
    }
}

bool CoreState::canRenameAny() const {
    std::lock_guard lock(mutex_);
    return !free_phys_regs_.empty();
}

bool CoreState::canAllocatePhysicalRegisters(std::size_t count) const {
    std::lock_guard lock(mutex_);
    return free_phys_regs_.size() >= count;
}

bool CoreState::canAllocateRob(std::size_t count) const {
    std::lock_guard lock(mutex_);
    return rob_capacity_ >= (rob_.size() - commit_head_) + count;
}

void CoreState::rename(DynInstPtr dyn_inst) {
    std::lock_guard lock(mutex_);
    if (!dyn_inst) {
        throw std::runtime_error("rename received a null dyninst");
    }

    const auto sequence = next_sequence_++;
    const auto& inst = dyn_inst->staticInst();
    const bool writes = writesRd(inst) && inst.rd != 0;

    if (writes && free_phys_regs_.empty()) {
        throw std::runtime_error("rename ran out of physical registers");
    }

    auto& rename = dyn_inst->renameState();
    rename.sequence = sequence;
    rename.writes_rd = writes;
    rename.load = isLoad(inst);
    rename.store = isStore(inst);
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

void CoreState::dispatchRenamed(DynInstPtr dyn_inst) {
    std::lock_guard lock(mutex_);
    if (!dyn_inst) {
        throw std::runtime_error("dispatch received a null dyninst");
    }

    auto& rename = dyn_inst->renameState();
    if (rename.dispatched) {
        throw std::runtime_error("dyninst was dispatched twice");
    }
    if (rename.sequence != rob_.size()) {
        std::ostringstream os;
        os << "dispatch sequence does not match rob tail: sequence="
           << rename.sequence << ", rob_tail=" << rob_.size();
        throw std::runtime_error(os.str());
    }
    if (rob_.size() - commit_head_ >= rob_capacity_) {
        throw std::runtime_error("dispatch ran out of rob entries");
    }

    rename.dispatched = true;
    rob_.push_back(RobEntry{dyn_inst});
}

void CoreState::discardRenamed(DynInstPtr dyn_inst) {
    std::lock_guard lock(mutex_);
    if (!dyn_inst) {
        return;
    }

    auto& rename = dyn_inst->renameState();
    if (rename.dispatched || rename.discarded) {
        return;
    }
    rename.discarded = true;
    if (rename.writes_rd) {
        free_phys_regs_.push_back(rename.phys_dst);
    }
    if (rename.sequence >= rob_.size()) {
        next_sequence_ = rob_.size();
    }
}

void CoreState::completeRedirectFlush() {
    std::lock_guard lock(mutex_);
    next_sequence_ = rob_.size();
    restoreSpeculativeRenameMapLocked();
}

void CoreState::markCompleted(DynInstPtr dyn_inst) {
    std::lock_guard lock(mutex_);
    if (!dyn_inst) {
        throw std::runtime_error("completion has no dyninst");
    }

    const auto sequence = dyn_inst->renameState().sequence;
    if (sequence >= rob_.size()) {
        throw std::runtime_error("completion references an unknown rob entry");
    }
    auto& entry = rob_[sequence];
    if (entry.inst != dyn_inst) {
        throw std::runtime_error("completion dyninst does not match rob entry");
    }
    entry.inst->commitState().completed = true;
}

RetireResult CoreState::retire(std::size_t max_count) {
    std::lock_guard lock(mutex_);

    RetireResult result;
    while (result.retired < max_count && commit_head_ < rob_.size()) {
        const auto& entry = rob_[commit_head_];
        if (!entry.inst->commitState().completed) {
            break;
        }

        const auto& inst = entry.inst->staticInst();
        const auto& rename = entry.inst->renameState();
        const auto& execute = entry.inst->executeState();
        const auto value =
            rename.writes_rd ? physical_regs_[rename.phys_dst] : std::uint64_t{0};
        result.retired_insts.push_back(entry.inst);
        result.trace.push_back(RetiredInstTrace{
            rename.sequence,
            entry.inst->pc(),
            inst.bits,
            inst.opcode,
            rename.writes_rd,
            inst.rd,
            value,
            execute.store,
            execute.redirect,
            execute.exception,
        });
        if (execute.exception) {
            if (rename.writes_rd) {
                free_phys_regs_.push_back(rename.phys_dst);
                auto& reg = registers_[static_cast<std::size_t>(inst.rd)];
                if (reg.producer && *reg.producer == rename.sequence) {
                    reg.producer.reset();
                }
            }
            result.redirect = ControlRedirect{entry.inst->pc() + 4, true};
            ++commit_head_;
            ++committed_;
            ++result.retired;
            flushAfterRedirectLocked(rename.sequence);
            break;
        }
        if (execute.store) {
            result.stores.push_back(RetiredStore{entry.inst, *execute.store});
        }
        if (execute.branch_update) {
            result.branch_updates.push_back(*execute.branch_update);
        }
        if (rename.writes_rd) {
            auto& reg = registers_[static_cast<std::size_t>(inst.rd)];
            free_phys_regs_.push_back(reg.committed_phys);
            reg.committed_phys = rename.phys_dst;
            if (reg.producer && *reg.producer == rename.sequence) {
                reg.producer.reset();
            }
        }

        ++commit_head_;
        ++committed_;
        ++result.retired;

        if (execute.redirect) {
            result.redirect = execute.redirect;
            flushAfterRedirectLocked(rename.sequence);
            break;
        }
    }
    return result;
}

std::size_t CoreState::committedCount() const {
    std::lock_guard lock(mutex_);
    return committed_;
}

std::size_t CoreState::inFlightCount() const {
    std::lock_guard lock(mutex_);
    return rob_.size() - commit_head_;
}

std::uint64_t CoreState::registerValue(int reg) const {
    std::lock_guard lock(mutex_);
    if (reg == 0) {
        return 0;
    }
    return physical_regs_[registers_[static_cast<std::size_t>(reg)].committed_phys];
}

std::size_t CoreState::freePhysicalRegisters() const {
    std::lock_guard lock(mutex_);
    return free_phys_regs_.size();
}

std::size_t CoreState::freeRobEntries() const {
    std::lock_guard lock(mutex_);
    return rob_capacity_ - (rob_.size() - commit_head_);
}

std::size_t CoreState::physicalRegisterCount() const noexcept {
    return physical_register_count_;
}

std::size_t CoreState::robCapacity() const noexcept {
    return rob_capacity_;
}

std::uint64_t CoreState::readPhysicalRegister(std::size_t phys) const noexcept {
    return physical_regs_[phys];
}

void CoreState::writePhysicalRegister(std::size_t phys,
                                      std::uint64_t value) noexcept {
    physical_regs_[phys] = value;
}

Operand CoreState::readSourceLocked(int reg) const {
    if (reg == 0) {
        return Operand{true, 0, 0};
    }

    const auto& state = registers_[static_cast<std::size_t>(reg)];
    if (state.producer) {
        if (*state.producer >= rob_.size()) {
            return Operand{false, state.phys, *state.producer};
        }
        const auto* producer = rob_.at(*state.producer).inst;
        if (producer->commitState().completed) {
            return Operand{true, state.phys, 0};
        }
        return Operand{false, state.phys, *state.producer};
    }
    return Operand{true, state.phys, 0};
}

void CoreState::flushAfterRedirectLocked(std::size_t redirect_sequence) {
    for (std::size_t i = redirect_sequence + 1; i < rob_.size(); ++i) {
        const auto* inst = rob_[i].inst;
        if (inst && inst->renameState().writes_rd) {
            free_phys_regs_.push_back(inst->renameState().phys_dst);
        }
    }

    rob_.resize(redirect_sequence + 1);
    commit_head_ = rob_.size();
    next_sequence_ = rob_.size();
    restoreSpeculativeRenameMapLocked();
}

void CoreState::restoreSpeculativeRenameMapLocked() {
    for (auto& reg : registers_) {
        reg.phys = reg.committed_phys;
        reg.producer.reset();
    }
}

} // namespace riscv_cpu
