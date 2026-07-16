#pragma once

#include "pipeline_types.hh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace riscv_cpu {

class BranchPredictor {
public:
    struct Stats {
        std::size_t predictions = 0;
        std::size_t conditional_predictions = 0;
        std::size_t unconditional_predictions = 0;
        std::size_t updates = 0;
        std::size_t conditional_updates = 0;
        std::size_t taken_updates = 0;
        std::size_t mispredictions = 0;
        std::size_t direction_mispredictions = 0;
        std::size_t target_mispredictions = 0;
        std::size_t micro_btb_lookups = 0;
        std::size_t micro_btb_hits = 0;
        std::size_t micro_btb_updates = 0;
        std::size_t micro_btb_allocations = 0;
        std::size_t tage_lookups = 0;
        std::size_t tage_provider_hits = 0;
        std::size_t tage_base_uses = 0;
        std::size_t tage_alternate_uses = 0;
        std::size_t tage_overrides = 0;
        std::size_t tage_allocations = 0;
        std::size_t tage_useful_ages = 0;
    };

    explicit BranchPredictor(Stats& stats);

    BranchPrediction predict(const StaticInst& inst, std::uint64_t pc);
    void update(const BranchUpdate& update);
    void update(const BranchUpdateBundle& updates);

private:
    static constexpr std::size_t kMicroBtbEntries = 128;
    static constexpr std::size_t kBimodalEntries = 512;
    static constexpr std::size_t kTaggedEntries = 256;
    static constexpr std::size_t kTageTables = TageSnapshot::table_count;
    static constexpr std::array<unsigned, kTageTables> kHistoryLengths{4, 8, 16, 32};
    static constexpr unsigned kTagBits = 12;

    struct MicroBtbEntry {
        bool valid = false;
        bool conditional = false;
        std::uint64_t tag = 0;
        std::uint64_t target = 0;
        std::uint8_t counter = 1;
    };

    struct TaggedEntry {
        bool valid = false;
        std::uint16_t tag = 0;
        std::int8_t counter = 0;
        std::uint8_t useful = 0;
    };

    struct MicroBtbPrediction {
        bool hit = false;
        bool taken = false;
        std::uint64_t target = 0;
        std::size_t index = 0;
        std::uint64_t tag = 0;
    };

    MicroBtbPrediction predictMicroBtb(const StaticInst& inst,
                                       std::uint64_t pc);
    TageSnapshot predictTage(std::uint64_t pc);
    void updateMicroBtb(const BranchUpdate& update);
    void updateTage(const BranchUpdate& update);

    static std::uint64_t branchTarget(std::uint64_t pc, const StaticInst& inst);
    static bool isConditionalBranch(const StaticInst& inst);
    static bool isDirectJump(const StaticInst& inst);
    static bool isIndirectJump(const StaticInst& inst);
    static bool predictsTaken(std::uint8_t counter) noexcept;
    static void updateCounter(std::uint8_t& counter, bool taken) noexcept;
    static bool predictsTaken(std::int8_t counter) noexcept;
    static void updateCounter(std::int8_t& counter, bool taken) noexcept;
    static std::uint64_t foldHistory(std::uint64_t history, unsigned length,
                                     unsigned width) noexcept;
    static std::uint16_t makeTag(std::uint64_t pc, std::uint64_t history,
                                 unsigned history_length) noexcept;
    static std::size_t makeIndex(std::uint64_t pc, std::uint64_t history,
                                 unsigned history_length,
                                 std::size_t table) noexcept;

    std::array<MicroBtbEntry, kMicroBtbEntries> micro_btb_{};
    std::array<std::uint8_t, kBimodalEntries> bimodal_{};
    std::array<std::array<TaggedEntry, kTaggedEntries>, kTageTables> tagged_{};
    Stats& stats_;
    std::uint64_t global_history_ = 0;
};

} // namespace riscv_cpu
