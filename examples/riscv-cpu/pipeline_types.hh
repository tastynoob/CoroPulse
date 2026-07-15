#pragma once

#include "isa.hh"
#include "types.hh"

#include <cstddef>
#include <cstdint>

namespace riscv_cpu {

class DynInst;
using DynInstPtr = DynInst*;

struct Operand {
    bool ready = true;
    std::uint64_t value = 0;
    std::size_t producer = 0;
};

struct RenameState {
    std::size_t sequence = 0;
    bool writes_rd = false;
    bool memory = false;
    std::size_t phys_dst = 0;
    std::size_t old_phys_dst = 0;
    Operand src1;
    Operand src2;
};

struct ExecuteState {
    std::uint64_t value = 0;
    coropulse::TickId done_tick = 0;
};

struct CommitState {
    bool completed = false;
};

struct ControlRedirect {
    std::uint64_t next_pc = 0;
    bool halt = false;
};

struct ExecResult {
    std::size_t sequence = 0;
    DynInstPtr inst = nullptr;
    std::uint64_t value = 0;
    bool writes_rd = false;
};

} // namespace riscv_cpu
