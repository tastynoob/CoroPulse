#include "stages.hh"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace riscv_cpu {

FetchStage::FetchStage(const SimpleSram& sram, DynInstPool& inst_pool,
                       std::size_t fetch_width)
    : sram_(sram),
      inst_pool_(inst_pool),
      fetch_width_(fetch_width),
      predictor_(stats.predictor) {
    if (fetch_width_ == 0) {
        throw std::runtime_error("fetch width must be greater than zero");
    }
}

coropulse::Task<void> FetchStage::process() {
    for (;; co_yield coropulse::tickDone{}) {
        applyPredictorUpdates();

        bool fifo_ready = true;
        if (auto redirect = redirect_in.read()) {
            applyRedirect(*redirect);
        } else {
            fifo_ready = co_await backend_can_accept.read();
        }

        (void)emitReadyPacket(fifo_ready);

        if (!halted_ && canBuildPacket()) {
            auto packet = buildPacket(pc_);
            if (packet.slots.empty()) {
                halted_ = true;
            } else {
                pc_ = packet.micro_next_pc;
                pipe_.push_back(std::move(packet));
            }
        }
    }
    co_return;
}

bool FetchStage::halted() const noexcept {
    return halted_;
}

bool FetchStage::architecturalHalted() const noexcept {
    return architectural_halted_;
}

void FetchStage::applyPredictorUpdates() {
    if (auto updates = predictor_update_in.take()) {
        predictor_.update(*updates);
    }
}

void FetchStage::applyRedirect(const ControlRedirect& redirect) {
    ++stats.redirects;
    pc_ = redirect.next_pc;
    pipe_.clear();
    halted_ = redirect.halt;
    architectural_halted_ = redirect.halt;
}

bool FetchStage::emitReadyPacket(bool fifo_ready) {
    if (pipe_.empty()) {
        return false;
    }

    auto& packet = pipe_.front();
    if (currentTick() < packet.tage_ready_tick) {
        return false;
    }

    const bool corrected = finalizePrediction(packet);
    if (corrected && !packet.correction_applied) {
        while (pipe_.size() > 1) {
            pipe_.pop_back();
        }
        pc_ = packet.final_next_pc;
        packet.correction_applied = true;
        ++stats.frontend_squashes;
    }

    if (!fifo_ready || !out.canWrite()) {
        if (!packet.slots.empty()) {
            ++stats.backpressure_stalls;
        }
        return false;
    }

    auto bundle = makeInstBundle(packet);
    if (!out.write(std::move(bundle))) {
        ++stats.backpressure_stalls;
        return false;
    }

    pipe_.pop_front();
    return true;
}

bool FetchStage::canBuildPacket() const {
    return pipe_.size() < 3;
}

FetchStage::FetchPacket FetchStage::buildPacket(std::uint64_t pc) {
    FetchPacket packet;
    packet.tage_ready_tick = currentTick() + 2;

    if (pc / 4 >= sram_.instructionCount()) {
        packet.micro_next_pc = pc;
        packet.final_next_pc = pc;
        return packet;
    }

    auto cursor = pc;
    while (packet.slots.size() < fetch_width_ &&
           cursor / 4 < sram_.instructionCount()) {
        const auto& inst = sram_.loadInstruction(cursor);
        FetchSlot slot;
        slot.inst = &inst;
        slot.pc = cursor;
        slot.predicted_next_pc = cursor + 4;

        packet.slots.push_back(slot);

        if (isControlFlow(inst)) {
            packet.has_control = true;
            packet.control_index = packet.slots.size() - 1;
            auto prediction = predictor_.predict(inst, cursor);
            auto& control_slot = packet.slots.back();
            control_slot.prediction = prediction;
            control_slot.predicted_next_pc = prediction.valid
                                                 ? prediction.predicted_next_pc
                                                 : cursor + 4;
            packet.micro_next_pc = control_slot.predicted_next_pc;
            packet.final_next_pc = packet.micro_next_pc;
            return packet;
        }

        cursor += 4;
    }

    packet.micro_next_pc = cursor;
    packet.final_next_pc = cursor;
    return packet;
}

bool FetchStage::finalizePrediction(FetchPacket& packet) {
    if (packet.finalized) {
        return packet.final_next_pc != packet.micro_next_pc;
    }

    packet.finalized = true;
    packet.final_next_pc = packet.micro_next_pc;
    if (!packet.has_control) {
        return false;
    }

    auto& slot = packet.slots[packet.control_index];
    auto& prediction = slot.prediction;
    if (!prediction.valid || !prediction.conditional || !prediction.tage.valid) {
        return false;
    }

    const bool tage_taken = prediction.tage.prediction;
    const auto tage_next_pc = tage_taken ? prediction.target_pc
                                         : prediction.fallthrough_pc;
    prediction.predicted_taken = tage_taken;
    prediction.predicted_next_pc = tage_next_pc;
    slot.predicted_next_pc = tage_next_pc;
    packet.final_next_pc = tage_next_pc;

    if (tage_next_pc != packet.micro_next_pc) {
        ++stats.predictor.tage_overrides;
        return true;
    }
    return false;
}

InstBundle FetchStage::makeInstBundle(const FetchPacket& packet) {
    InstBundle bundle;
    bundle.reserve(packet.slots.size());
    for (const auto& slot : packet.slots) {
        bundle.push_back(inst_pool_.create(*slot.inst, slot.pc,
                                           slot.predicted_next_pc,
                                           slot.prediction));
        ++stats.fetched;
    }
    return bundle;
}

} // namespace riscv_cpu
