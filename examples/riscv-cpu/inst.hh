#pragma once

#include "pipeline_types.hh"

#include <cstdint>
#include <deque>

namespace riscv_cpu {

class DynInst {
public:
    DynInst(const StaticInst& static_inst, std::uint64_t pc,
            std::uint64_t fetch_sequence);

    const StaticInst& staticInst() const noexcept;
    std::uint64_t pc() const noexcept;
    std::uint64_t fetchSequence() const noexcept;

    RenameState& renameState() noexcept;
    const RenameState& renameState() const noexcept;
    ExecuteState& executeState() noexcept;
    const ExecuteState& executeState() const noexcept;
    CommitState& commitState() noexcept;
    const CommitState& commitState() const noexcept;

private:
    const StaticInst* static_inst_ = nullptr;
    std::uint64_t pc_ = 0;
    std::uint64_t fetch_sequence_ = 0;
    RenameState rename_;
    ExecuteState execute_;
    CommitState commit_;
};

class DynInstPool {
public:
    DynInstPtr create(const StaticInst& static_inst, std::uint64_t pc);

private:
    std::deque<DynInst> insts_;
    std::uint64_t next_fetch_sequence_ = 0;
};

} // namespace riscv_cpu
