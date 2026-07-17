#pragma once

#include "isa.hh"
#include "types.hh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace riscv_cpu {

class DynInst;
using DynInstPtr = DynInst*;
using InstBundle = std::vector<DynInstPtr>;

struct Operand {
    bool ready = true;
    std::size_t phys = 0;
    std::size_t producer = 0;
};

struct RenameState {
    static constexpr std::size_t no_lsq_index =
        static_cast<std::size_t>(-1);

    std::size_t sequence = 0;
    bool writes_rd = false;
    bool load = false;
    bool store = false;
    bool memory = false;
    bool dispatched = false;
    bool discarded = false;
    std::size_t phys_dst = 0;
    std::size_t old_phys_dst = 0;
    std::size_t load_queue_index = no_lsq_index;
    std::size_t store_queue_index = no_lsq_index;
    Operand src1;
    Operand src2;
};

struct ControlRedirect {
    std::uint64_t next_pc = 0;
    bool halt = false;
};

struct TageSnapshot {
    static constexpr std::size_t table_count = 4;

    bool valid = false;
    bool prediction = false;
    bool base_prediction = false;
    bool alternate_prediction = false;
    int provider_table = -1;
    int alternate_table = -1;
    std::size_t base_index = 0;
    std::array<std::size_t, table_count> indices{};
    std::array<std::uint16_t, table_count> tags{};
};

struct BranchPrediction {
    bool valid = false;
    bool conditional = false;
    bool predicted_taken = false;
    bool micro_btb_hit = false;
    std::uint64_t pc = 0;
    std::uint64_t fallthrough_pc = 0;
    std::uint64_t target_pc = 0;
    std::uint64_t predicted_next_pc = 0;
    std::size_t micro_btb_index = 0;
    std::uint64_t micro_btb_tag = 0;
    TageSnapshot tage;
};

struct BranchUpdate {
    bool valid = false;
    bool conditional = false;
    bool taken = false;
    bool mispredicted = false;
    std::uint64_t pc = 0;
    std::uint64_t target_pc = 0;
    std::uint64_t actual_next_pc = 0;
    std::uint64_t predicted_next_pc = 0;
    BranchPrediction prediction;
};

using BranchUpdateBundle = std::vector<BranchUpdate>;

struct StoreWrite {
    std::uint64_t address = 0;
    std::uint64_t value = 0;
    std::size_t bytes = 0;
};

enum class ExceptionCode {
    load_access_fault,
    store_access_fault,
};

struct ExceptionState {
    ExceptionCode code = ExceptionCode::load_access_fault;
    std::uint64_t fault_address = 0;
    std::string message;
};

struct ExecuteState {
    coropulse::TickId done_tick = 0;
    std::optional<ControlRedirect> redirect;
    std::optional<BranchUpdate> branch_update;
    std::optional<StoreWrite> store;
    std::optional<ExceptionState> exception;
};

struct CommitState {
    bool completed = false;
};

struct RetiredInstTrace {
    std::size_t sequence = 0;
    std::uint64_t pc = 0;
    std::uint32_t bits = 0;
    Opcode opcode = Opcode::illegal;
    bool writes_rd = false;
    int rd = 0;
    std::uint64_t value = 0;
    std::optional<StoreWrite> store;
    std::optional<ControlRedirect> redirect;
    std::optional<ExceptionState> exception;
};

struct RetiredStore {
    DynInstPtr inst = nullptr;
    StoreWrite store;
};

struct RetireResult {
    std::size_t retired = 0;
    InstBundle retired_insts;
    std::vector<RetiredStore> stores;
    BranchUpdateBundle branch_updates;
    std::vector<RetiredInstTrace> trace;
    std::optional<ControlRedirect> redirect;
};

} // namespace riscv_cpu
