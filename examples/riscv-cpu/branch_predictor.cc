#include "branch_predictor.hh"

#include <algorithm>
#include <cstdlib>

namespace riscv_cpu {
namespace {

std::uint64_t addSigned(std::uint64_t value, std::int64_t offset) {
    return static_cast<std::uint64_t>(static_cast<std::int64_t>(value) + offset);
}

} // namespace

BranchPredictor::BranchPredictor(Stats& stats) : stats_(stats) {
    bimodal_.fill(1);
}

BranchPrediction BranchPredictor::predict(const StaticInst& inst,
                                          std::uint64_t pc) {
    BranchPrediction prediction;
    if (!isControlFlow(inst) || isHalt(inst)) {
        return prediction;
    }

    ++stats_.predictions;
    const auto fallthrough = pc + 4;
    const auto micro = predictMicroBtb(inst, pc);
    prediction.valid = true;
    prediction.conditional = isConditionalBranch(inst);
    if (prediction.conditional) {
        ++stats_.conditional_predictions;
    } else {
        ++stats_.unconditional_predictions;
    }
    prediction.pc = pc;
    prediction.fallthrough_pc = fallthrough;
    prediction.target_pc = micro.target;
    prediction.predicted_taken = micro.taken;
    prediction.predicted_next_pc = micro.taken ? micro.target : fallthrough;
    prediction.micro_btb_hit = micro.hit;
    prediction.micro_btb_index = micro.index;
    prediction.micro_btb_tag = micro.tag;

    if (prediction.conditional) {
        prediction.tage = predictTage(pc);
    }
    return prediction;
}

void BranchPredictor::update(const BranchUpdate& update) {
    if (!update.valid) {
        return;
    }

    ++stats_.updates;
    if (update.conditional) {
        ++stats_.conditional_updates;
    }
    if (update.taken) {
        ++stats_.taken_updates;
    }
    if (update.mispredicted) {
        ++stats_.mispredictions;
        if (update.prediction.predicted_taken != update.taken) {
            ++stats_.direction_mispredictions;
        } else {
            ++stats_.target_mispredictions;
        }
    }

    updateMicroBtb(update);
    if (update.conditional) {
        updateTage(update);
        global_history_ = ((global_history_ << 1) | (update.taken ? 1ULL : 0ULL));
    }
}

void BranchPredictor::update(const BranchUpdateBundle& updates) {
    for (const auto& item : updates) {
        update(item);
    }
}

BranchPredictor::MicroBtbPrediction
BranchPredictor::predictMicroBtb(const StaticInst& inst, std::uint64_t pc) {
    ++stats_.micro_btb_lookups;
    const auto index = static_cast<std::size_t>((pc >> 2) & (kMicroBtbEntries - 1));
    const auto tag = pc >> 2;
    const auto& entry = micro_btb_[index];

    MicroBtbPrediction prediction;
    prediction.index = index;
    prediction.tag = tag;
    prediction.target = branchTarget(pc, inst);

    if (!entry.valid || entry.tag != tag) {
        if (isDirectJump(inst)) {
            prediction.taken = true;
        }
        return prediction;
    }

    prediction.hit = true;
    ++stats_.micro_btb_hits;
    prediction.target = entry.target;
    prediction.taken = !entry.conditional || predictsTaken(entry.counter);
    return prediction;
}

TageSnapshot BranchPredictor::predictTage(std::uint64_t pc) {
    ++stats_.tage_lookups;
    TageSnapshot snapshot;
    snapshot.valid = true;
    snapshot.base_index =
        static_cast<std::size_t>(((pc >> 2) ^ foldHistory(global_history_, 16, 9)) &
                                 (kBimodalEntries - 1));
    snapshot.base_prediction = predictsTaken(bimodal_[snapshot.base_index]);
    snapshot.alternate_prediction = snapshot.base_prediction;

    for (std::size_t table = 0; table < kTageTables; ++table) {
        snapshot.indices[table] =
            makeIndex(pc, global_history_, kHistoryLengths[table], table);
        snapshot.tags[table] =
            makeTag(pc, global_history_, kHistoryLengths[table]);
    }

    for (int table = static_cast<int>(kTageTables) - 1; table >= 0; --table) {
        const auto& entry =
            tagged_[static_cast<std::size_t>(table)]
                   [snapshot.indices[static_cast<std::size_t>(table)]];
        if (entry.valid &&
            entry.tag == snapshot.tags[static_cast<std::size_t>(table)]) {
            if (snapshot.provider_table < 0) {
                snapshot.provider_table = table;
                snapshot.prediction = predictsTaken(entry.counter);
            } else {
                snapshot.alternate_table = table;
                snapshot.alternate_prediction = predictsTaken(entry.counter);
                break;
            }
        }
    }

    if (snapshot.provider_table < 0) {
        ++stats_.tage_base_uses;
        snapshot.prediction = snapshot.base_prediction;
    } else if (snapshot.alternate_table < 0) {
        ++stats_.tage_provider_hits;
        snapshot.alternate_prediction = snapshot.base_prediction;
    } else {
        ++stats_.tage_provider_hits;
    }

    if (snapshot.provider_table >= 0) {
        const auto table = static_cast<std::size_t>(snapshot.provider_table);
        const auto& entry = tagged_[table][snapshot.indices[table]];
        const bool weak_provider = entry.counter >= -1 && entry.counter <= 0;
        if (weak_provider && entry.useful == 0) {
            snapshot.prediction = snapshot.alternate_prediction;
            ++stats_.tage_alternate_uses;
        }
    }

    return snapshot;
}

void BranchPredictor::updateMicroBtb(const BranchUpdate& update) {
    ++stats_.micro_btb_updates;
    const auto index =
        static_cast<std::size_t>((update.pc >> 2) & (kMicroBtbEntries - 1));
    const auto tag = update.pc >> 2;
    auto& entry = micro_btb_[index];

    bool updated_counter = false;
    if (entry.valid && entry.tag == tag && entry.conditional) {
        updateCounter(entry.counter, update.taken);
        updated_counter = true;
    }

    if (!update.taken) {
        return;
    }

    if (!entry.valid || entry.tag != tag) {
        entry = MicroBtbEntry{};
        entry.valid = true;
        entry.tag = tag;
        entry.counter = update.taken ? 2 : 1;
        ++stats_.micro_btb_allocations;
    }

    entry.conditional = update.conditional;
    entry.target = update.target_pc;
    if (entry.conditional) {
        if (!updated_counter) {
            updateCounter(entry.counter, update.taken);
        }
    } else {
        entry.counter = 3;
    }
}

void BranchPredictor::updateTage(const BranchUpdate& update) {
    const auto& meta = update.prediction.tage;
    if (!meta.valid) {
        return;
    }

    updateCounter(bimodal_[meta.base_index], update.taken);

    int provider = meta.provider_table;
    bool provider_updated = false;
    if (provider >= 0) {
        const auto table = static_cast<std::size_t>(provider);
        auto& entry = tagged_[table][meta.indices[table]];
        if (entry.valid && entry.tag == meta.tags[table]) {
            const bool provider_prediction = predictsTaken(entry.counter);
            updateCounter(entry.counter, update.taken);
            provider_updated = true;

            if (provider_prediction != meta.alternate_prediction) {
                if (provider_prediction == update.taken) {
                    entry.useful = std::min<std::uint8_t>(3, entry.useful + 1);
                } else if (entry.useful > 0) {
                    --entry.useful;
                }
            }
        }
    }

    if (!update.mispredicted) {
        return;
    }

    const auto start_table = static_cast<std::size_t>(provider + 1);
    bool allocated = false;
    for (std::size_t table = start_table; table < kTageTables; ++table) {
        auto& entry = tagged_[table][meta.indices[table]];
        if (!entry.valid || entry.useful == 0) {
            entry.valid = true;
            entry.tag = meta.tags[table];
            entry.counter = update.taken ? 0 : -1;
            entry.useful = 0;
            ++stats_.tage_allocations;
            allocated = true;
            break;
        }
    }

    if (!allocated && provider_updated) {
        for (std::size_t table = start_table; table < kTageTables; ++table) {
            auto& entry = tagged_[table][meta.indices[table]];
            if (entry.useful > 0) {
                --entry.useful;
                ++stats_.tage_useful_ages;
            }
        }
    }
}

std::uint64_t BranchPredictor::branchTarget(std::uint64_t pc,
                                            const StaticInst& inst) {
    if (isConditionalBranch(inst) || isDirectJump(inst)) {
        return addSigned(pc, inst.imm);
    }
    return pc + 4;
}

bool BranchPredictor::isConditionalBranch(const StaticInst& inst) {
    switch (inst.opcode) {
    case Opcode::beq:
    case Opcode::bne:
    case Opcode::blt:
    case Opcode::bge:
    case Opcode::bltu:
    case Opcode::bgeu:
        return true;
    default:
        return false;
    }
}

bool BranchPredictor::isDirectJump(const StaticInst& inst) {
    return inst.opcode == Opcode::jal;
}

bool BranchPredictor::isIndirectJump(const StaticInst& inst) {
    return inst.opcode == Opcode::jalr;
}

bool BranchPredictor::predictsTaken(std::uint8_t counter) noexcept {
    return counter >= 2;
}

void BranchPredictor::updateCounter(std::uint8_t& counter, bool taken) noexcept {
    if (taken) {
        counter = std::min<std::uint8_t>(3, counter + 1);
    } else if (counter > 0) {
        --counter;
    }
}

bool BranchPredictor::predictsTaken(std::int8_t counter) noexcept {
    return counter >= 0;
}

void BranchPredictor::updateCounter(std::int8_t& counter, bool taken) noexcept {
    if (taken) {
        counter = std::min<std::int8_t>(3, counter + 1);
    } else {
        counter = std::max<std::int8_t>(-4, counter - 1);
    }
}

std::uint64_t BranchPredictor::foldHistory(std::uint64_t history,
                                           unsigned length,
                                           unsigned width) noexcept {
    if (width == 0) {
        return 0;
    }

    const auto mask = (std::uint64_t{1} << width) - 1;
    std::uint64_t folded = 0;
    for (unsigned offset = 0; offset < length; offset += width) {
        folded ^= (history >> offset) & mask;
    }
    return folded & mask;
}

std::uint16_t BranchPredictor::makeTag(std::uint64_t pc,
                                       std::uint64_t history,
                                       unsigned history_length) noexcept {
    const auto folded = foldHistory(history, history_length, kTagBits);
    const auto mixed = (pc >> 2) ^ folded ^ (pc >> 17);
    return static_cast<std::uint16_t>(mixed & ((1U << kTagBits) - 1));
}

std::size_t BranchPredictor::makeIndex(std::uint64_t pc,
                                       std::uint64_t history,
                                       unsigned history_length,
                                       std::size_t table) noexcept {
    const auto folded = foldHistory(history, history_length, 8);
    const auto mixed = (pc >> 2) ^ folded ^ (pc >> (5 + table));
    return static_cast<std::size_t>(mixed & (kTaggedEntries - 1));
}

} // namespace riscv_cpu
