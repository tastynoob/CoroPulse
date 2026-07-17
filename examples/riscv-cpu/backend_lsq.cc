#include "backend_pipeline.hh"

#include "exec_context.hh"

#include <algorithm>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace riscv_cpu {
namespace {

std::uint64_t addSignedImmediate(std::uint64_t value, std::int64_t imm) {
    return static_cast<std::uint64_t>(static_cast<std::int64_t>(value) + imm);
}

std::uint64_t signExtend(std::uint64_t value, std::size_t bits) {
    const auto sign_bit = std::uint64_t{1} << (bits - 1);
    const auto mask = bits == 64 ? ~std::uint64_t{0}
                                 : ((std::uint64_t{1} << bits) - 1);
    value &= mask;
    return (value ^ sign_bit) - sign_bit;
}

} // namespace

LoadStoreQueue::LoadStoreQueue(std::size_t load_capacity,
                               std::size_t store_capacity)
    : load_capacity_(load_capacity), store_capacity_(store_capacity) {
    if (load_capacity_ == 0 || store_capacity_ == 0) {
        throw std::runtime_error("load/store queue capacities must be greater than zero");
    }
}

bool LoadStoreQueue::canAccept(const InstBundle& bundle) const {
    std::size_t loads = 0;
    std::size_t stores = 0;
    for (const auto* inst : bundle) {
        if (isLoad(inst->staticInst())) {
            ++loads;
        } else if (isStore(inst->staticInst())) {
            ++stores;
        }
    }
    return freeLoadEntries() >= loads && freeStoreEntries() >= stores;
}

std::size_t LoadStoreQueue::freeLoadEntries() const {
    return load_capacity_ - (loads_.size() - load_head_);
}

std::size_t LoadStoreQueue::freeStoreEntries() const {
    return store_capacity_ - (stores_.size() - store_head_);
}

void LoadStoreQueue::acceptRenamed(const InstBundle& bundle, BackendStats& stats) {
    if (!canAccept(bundle)) {
        throw std::runtime_error("lsq accepted a bundle while full");
    }

    for (auto* inst : bundle) {
        auto& rename = inst->renameState();
        if (rename.load) {
            rename.load_queue_index = loads_.size();
            loads_.push_back(LoadEntry{inst, false});
        } else if (rename.store) {
            rename.store_queue_index = stores_.size();
            stores_.push_back(StoreEntry{inst, false, {}});
        }
    }

    stats.load_queue_max_occupancy =
        std::max(stats.load_queue_max_occupancy, loads_.size() - load_head_);
    stats.store_queue_max_occupancy =
        std::max(stats.store_queue_max_occupancy, stores_.size() - store_head_);
}

LoadStoreQueue::LoadResult LoadStoreQueue::executeLoad(
    DynInstPtr inst, std::uint64_t address, std::size_t bytes,
    bool sign_extend, SimpleSram& sram, BackendStats& stats) {
    auto& load = loadEntry(inst);
    const auto load_sequence = inst->renameState().sequence;
    const StoreEntry* forwarding_store = nullptr;

    for (std::size_t i = store_head_; i < stores_.size(); ++i) {
        const auto& store = stores_[i];
        if (!store.inst ||
            store.inst->renameState().sequence >= load_sequence) {
            break;
        }
        if (!store.ready) {
            ++stats.load_store_waits;
            return {};
        }
        if (!overlaps(address, bytes, store.store.address, store.store.bytes)) {
            continue;
        }
        if (!covers(store.store, address, bytes)) {
            ++stats.load_store_waits;
            return {};
        }
        forwarding_store = &store;
    }

    load.completed = true;
    if (forwarding_store) {
        ++stats.load_store_forwards;
        return LoadResult{
            true,
            true,
            forwardValue(forwarding_store->store, address, bytes, sign_extend),
        };
    }

    return LoadResult{true, false, sram.load(address, bytes, sign_extend)};
}

void LoadStoreQueue::markStoreReady(DynInstPtr inst, StoreWrite store) {
    auto& entry = storeEntry(inst);
    entry.ready = true;
    entry.store = store;
}

void LoadStoreQueue::retire(DynInstPtr inst) {
    if (!inst || !inst->renameState().memory) {
        return;
    }

    const auto& rename = inst->renameState();
    if (rename.load) {
        if (load_head_ >= loads_.size() || loads_[load_head_].inst != inst) {
            throw std::runtime_error("load retired out of load queue order");
        }
        ++load_head_;
    } else if (rename.store) {
        if (store_head_ >= stores_.size() || stores_[store_head_].inst != inst) {
            throw std::runtime_error("store retired out of store queue order");
        }
        ++store_head_;
    }

    compact();
}

void LoadStoreQueue::clear() {
    loads_.clear();
    stores_.clear();
    load_head_ = 0;
    store_head_ = 0;
}

bool LoadStoreQueue::overlaps(std::uint64_t lhs_address, std::size_t lhs_bytes,
                              std::uint64_t rhs_address, std::size_t rhs_bytes) {
    const auto lhs_end = lhs_address + lhs_bytes;
    const auto rhs_end = rhs_address + rhs_bytes;
    return lhs_address < rhs_end && rhs_address < lhs_end;
}

bool LoadStoreQueue::covers(const StoreWrite& store, std::uint64_t address,
                            std::size_t bytes) {
    return store.address <= address && address + bytes <= store.address + store.bytes;
}

std::uint64_t LoadStoreQueue::forwardValue(const StoreWrite& store,
                                           std::uint64_t address,
                                           std::size_t bytes,
                                           bool sign_extend) {
    const auto shift = static_cast<unsigned>((address - store.address) * 8);
    auto value = store.value >> shift;
    if (bytes < 8) {
        value &= (std::uint64_t{1} << (bytes * 8)) - 1;
    }
    if (sign_extend && bytes < 8) {
        return signExtend(value, bytes * 8);
    }
    return value;
}

LoadStoreQueue::LoadEntry& LoadStoreQueue::loadEntry(DynInstPtr inst) {
    const auto index = inst->renameState().load_queue_index;
    if (index >= loads_.size() || loads_[index].inst != inst) {
        throw std::runtime_error("load does not match load queue entry");
    }
    return loads_[index];
}

LoadStoreQueue::StoreEntry& LoadStoreQueue::storeEntry(DynInstPtr inst) {
    const auto index = inst->renameState().store_queue_index;
    if (index >= stores_.size() || stores_[index].inst != inst) {
        throw std::runtime_error("store does not match store queue entry");
    }
    return stores_[index];
}

void LoadStoreQueue::compact() {
    if (load_head_ == loads_.size()) {
        loads_.clear();
        load_head_ = 0;
    }
    if (store_head_ == stores_.size()) {
        stores_.clear();
        store_head_ = 0;
    }
}

LoadStorePipe::LoadStorePipe(CoreState& core, SimpleSram& sram,
                             LoadStoreQueue& lsq, std::size_t memory_width)
    : core_(core),
      sram_(sram),
      lsq_(lsq),
      memory_width_(memory_width) {
    if (memory_width_ == 0) {
        throw std::runtime_error("memory width must be greater than zero");
    }
}

void LoadStorePipe::accept(InstBundle&& bundle, coropulse::TickId tick,
                           BackendStats& stats) {
    for (auto* inst : bundle) {
        inst->executeState().done_tick = tick + inst->staticInst().latency;
        executing_.push_back(Executing{
            inst,
            inst->executeState().done_tick,
        });
    }
    stats.memory_execute_accepted += bundle.size();
}

InstBundle LoadStorePipe::collectCompletions(coropulse::TickId tick,
                                             BackendStats& stats) {
    InstBundle completed;
    if (!pending_completion_.empty()) {
        completed = std::move(pending_completion_);
        pending_completion_.clear();
        stats.memory_execute_completed += completed.size();
    }

    completeReadyUops(tick, stats);
    return completed;
}

void LoadStorePipe::clear() {
    executing_.clear();
    pending_completion_.clear();
}

bool LoadStorePipe::execute(DynInstPtr inst, coropulse::TickId tick,
                            BackendStats& stats) {
    ExecContext context(core_, sram_, *inst, tick);
    if (isLoad(inst->staticInst())) {
        return executeLoad(context, inst, stats);
    }
    if (isStore(inst->staticInst())) {
        return executeStore(context, inst);
    }
    throw std::runtime_error("load/store pipe received a non-memory instruction");
}

bool LoadStorePipe::executeLoad(ExecContext& context, DynInstPtr inst,
                                BackendStats& stats) {
    const auto& static_inst = inst->staticInst();
    const auto address = addSignedImmediate(context.readSrc1(), static_inst.imm);
    const auto bytes = memoryAccessBytes(static_inst);
    const auto sign_extend = loadSignExtends(static_inst);

    try {
        auto result =
            lsq_.executeLoad(inst, address, bytes, sign_extend, sram_, stats);
        if (!result.completed) {
            return false;
        }
        context.writeResult(result.value);
        return true;
    } catch (const std::exception& error) {
        context.setException(ExceptionCode::load_access_fault, address,
                             error.what());
        return true;
    }
}

bool LoadStorePipe::executeStore(ExecContext& context, DynInstPtr inst) {
    const auto& static_inst = inst->staticInst();
    const auto address = addSignedImmediate(context.readSrc1(), static_inst.imm);
    const auto bytes = memoryAccessBytes(static_inst);
    const auto value = context.readSrc2();

    try {
        context.validateStore(address, bytes);
        context.setStore(address, value, bytes);
        lsq_.markStoreReady(inst, StoreWrite{address, value, bytes});
        return true;
    } catch (const std::exception& error) {
        context.setException(ExceptionCode::store_access_fault, address,
                             error.what());
        lsq_.markStoreReady(inst, StoreWrite{address, value, bytes});
        return true;
    }
}

void LoadStorePipe::completeReadyUops(coropulse::TickId tick,
                                      BackendStats& stats) {
    if (!pending_completion_.empty()) {
        return;
    }

    std::size_t completed_this_tick = 0;
    for (auto iter = executing_.begin(); iter != executing_.end();) {
        if (completed_this_tick >= memory_width_) {
            break;
        }
        if (iter->done_tick > tick) {
            ++iter;
            continue;
        }
        if (execute(iter->inst, tick, stats)) {
            pending_completion_.push_back(iter->inst);
            iter = executing_.erase(iter);
            ++completed_this_tick;
        } else {
            iter->done_tick = tick + 1;
            ++iter;
        }
    }
}

} // namespace riscv_cpu
