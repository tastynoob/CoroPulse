#pragma once

#include "core_state.hh"
#include "inst.hh"
#include "memory.hh"
#include "sim.hh"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <iosfwd>
#include <optional>
#include <vector>

namespace riscv_cpu {

class ExecContext;

struct BackendStats {
    std::size_t decoded = 0;
    std::size_t decode_backpressure_stalls = 0;
    std::size_t renamed = 0;
    std::size_t resource_stalls = 0;
    std::size_t issue_backpressure_stalls = 0;
    std::size_t rob_resource_stalls = 0;
    std::size_t lsq_resource_stalls = 0;
    std::size_t accepted = 0;
    std::size_t issued = 0;
    std::size_t memory_issued = 0;
    std::size_t execute_accepted = 0;
    std::size_t execute_completed = 0;
    std::size_t memory_execute_accepted = 0;
    std::size_t memory_execute_completed = 0;
    std::size_t load_store_forwards = 0;
    std::size_t load_store_waits = 0;
    std::size_t load_queue_max_occupancy = 0;
    std::size_t store_queue_max_occupancy = 0;
    std::size_t retired = 0;
    std::size_t redirects = 0;
    std::size_t frontend_queue_max_occupancy = 0;
    std::size_t frontend_queue_stalls = 0;
};

class FrontendQueue {
public:
    FrontendQueue(std::size_t capacity, std::size_t fetch_width,
                  std::size_t decode_width);

    bool canAcceptFetch() const;
    bool hasDecodeBundle() const noexcept;
    void acceptFetch(InstBundle&& bundle, BackendStats& stats);
    InstBundle popDecodeBundle();
    void clear() noexcept;

private:
    std::size_t capacity_;
    std::size_t fetch_width_;
    std::size_t decode_width_;
    std::deque<DynInstPtr> queue_;
};

class DecodePipe {
public:
    bool slotOpen() const noexcept;
    bool hasBundle() const noexcept;
    const InstBundle& bundle() const;
    InstBundle takeBundle();
    void accept(InstBundle&& bundle, BackendStats& stats);
    void clear() noexcept;

private:
    std::optional<InstBundle> reg_;
};

class RenamePipe {
public:
    bool slotOpen() const noexcept;
    bool hasBundle() const noexcept;
    const InstBundle& bundle() const;
    InstBundle takeBundle();
    void accept(CoreState& core, InstBundle&& bundle, BackendStats& stats);
    void clear() noexcept;

private:
    std::optional<InstBundle> reg_;
};

class IssuePipe {
public:
    IssuePipe(CoreState& core, std::size_t capacity, std::size_t dispatch_width,
              std::size_t issue_width);

    bool canAccept(std::size_t count) const;
    void rememberCompletions(const InstBundle& completions);
    void acceptRenamed(const InstBundle& bundle, BackendStats& stats);
    InstBundle issue(BackendStats& stats);
    void clear();

private:
    struct Entry {
        DynInstPtr inst = nullptr;
    };

    bool operandsReady(const Entry& entry) const;
    bool canIssue(const Entry& entry) const;
    std::vector<std::size_t> findReadyBundle() const;
    Entry makeEntry(DynInstPtr inst) const;
    void rememberCompletion(DynInstPtr completion);
    void applyWakeup(DynInstPtr completion);
    void applyKnownWakeups(Entry& entry) const;
    void applyKnownWakeup(Operand& operand) const;
    static void wakeOperand(Operand& operand, DynInstPtr producer);

    CoreState& core_;
    std::size_t capacity_;
    std::size_t dispatch_width_;
    std::size_t issue_width_;
    std::vector<Entry> queue_;
    std::vector<char> completed_;
};

class LoadStoreQueue {
public:
    struct LoadResult {
        bool completed = false;
        bool forwarded = false;
        std::uint64_t value = 0;
    };

    LoadStoreQueue(std::size_t load_capacity, std::size_t store_capacity);

    bool canAccept(const InstBundle& bundle) const;
    std::size_t freeLoadEntries() const;
    std::size_t freeStoreEntries() const;
    void acceptRenamed(const InstBundle& bundle, BackendStats& stats);
    LoadResult executeLoad(DynInstPtr inst, std::uint64_t address,
                           std::size_t bytes, bool sign_extend,
                           SimpleSram& sram, BackendStats& stats);
    void markStoreReady(DynInstPtr inst, StoreWrite store);
    void retire(DynInstPtr inst);
    void clear();

private:
    struct LoadEntry {
        DynInstPtr inst = nullptr;
        bool completed = false;
    };

    struct StoreEntry {
        DynInstPtr inst = nullptr;
        bool ready = false;
        StoreWrite store;
    };

    static bool overlaps(std::uint64_t lhs_address, std::size_t lhs_bytes,
                         std::uint64_t rhs_address, std::size_t rhs_bytes);
    static bool covers(const StoreWrite& store, std::uint64_t address,
                       std::size_t bytes);
    static std::uint64_t forwardValue(const StoreWrite& store,
                                      std::uint64_t address,
                                      std::size_t bytes,
                                      bool sign_extend);

    LoadEntry& loadEntry(DynInstPtr inst);
    StoreEntry& storeEntry(DynInstPtr inst);
    void compact();

    std::size_t load_capacity_;
    std::size_t store_capacity_;
    std::vector<LoadEntry> loads_;
    std::vector<StoreEntry> stores_;
    std::size_t load_head_ = 0;
    std::size_t store_head_ = 0;
};

class ExecutePipe {
public:
    ExecutePipe(CoreState& core, SimpleSram& sram);

    void accept(InstBundle&& bundle, coropulse::TickId tick, BackendStats& stats);
    InstBundle collectCompletions(coropulse::TickId tick, BackendStats& stats);
    void clear();

private:
    struct Executing {
        DynInstPtr inst = nullptr;
        coropulse::TickId done_tick = 0;
    };

    void execute(DynInstPtr inst, coropulse::TickId tick);
    void completeReadyUops(coropulse::TickId tick);

    CoreState& core_;
    SimpleSram& sram_;
    std::vector<Executing> executing_;
    InstBundle pending_completion_;
};

class LoadStorePipe {
public:
    LoadStorePipe(CoreState& core, SimpleSram& sram, LoadStoreQueue& lsq,
                  std::size_t memory_width);

    void accept(InstBundle&& bundle, coropulse::TickId tick, BackendStats& stats);
    InstBundle collectCompletions(coropulse::TickId tick, BackendStats& stats);
    void clear();

private:
    struct Executing {
        DynInstPtr inst = nullptr;
        coropulse::TickId done_tick = 0;
    };

    bool execute(DynInstPtr inst, coropulse::TickId tick, BackendStats& stats);
    bool executeLoad(ExecContext& context, DynInstPtr inst,
                     BackendStats& stats);
    bool executeStore(ExecContext& context, DynInstPtr inst);
    void completeReadyUops(coropulse::TickId tick, BackendStats& stats);

    CoreState& core_;
    SimpleSram& sram_;
    LoadStoreQueue& lsq_;
    std::size_t memory_width_;
    std::vector<Executing> executing_;
    InstBundle pending_completion_;
};

class CommitPipe {
public:
    CommitPipe(CoreState& core, SimpleSram& sram, LoadStoreQueue& lsq,
               std::size_t commit_width, std::ostream* trace_out,
               std::size_t trace_limit);

    bool beginTick();
    bool hasPendingRedirect() const noexcept;
    bool publishPendingRedirect(coropulse::Output<ControlRedirect>& output,
                                BackendStats& stats);
    bool publishPredictorUpdates(coropulse::Output<BranchUpdateBundle>& output);
    void queuePredictorUpdates(BranchUpdateBundle&& updates);
    RetireResult retire(coropulse::TickId tick, BackendStats& stats);
    void markCompleted(const InstBundle& bundle);
    void dispatch(const InstBundle& bundle);
    void discard(const InstBundle* bundle);
    void completeRedirectFlush();

private:
    void traceRetired(coropulse::TickId tick, const RetiredInstTrace& inst);

    CoreState& core_;
    SimpleSram& sram_;
    LoadStoreQueue& lsq_;
    std::size_t commit_width_;
    std::ostream* trace_out_ = nullptr;
    std::size_t trace_limit_ = 0;
    std::size_t trace_count_ = 0;
    std::optional<ControlRedirect> pending_redirect_;
    BranchUpdateBundle pending_updates_;
    bool flush_next_tick_ = false;
};

class BackendPipeline {
public:
    BackendPipeline(CoreState& core, SimpleSram& sram, std::size_t rename_width,
                    std::size_t frontend_queue_capacity,
                    std::size_t fetch_width, std::size_t decode_width,
                    std::size_t issue_capacity,
                    std::size_t load_queue_capacity,
                    std::size_t store_queue_capacity,
                    std::size_t dispatch_width, std::size_t issue_width,
                    std::size_t memory_width, std::size_t commit_width,
                    std::ostream* trace_out, std::size_t trace_limit);

    std::size_t renameWidth() const noexcept;
    void flushAfterRedirect(const InstBundle* dispatch,
                            coropulse::Input<InstBundle>& input);

    FrontendQueue frontend_queue;
    DecodePipe decode;
    RenamePipe rename;
    IssuePipe issue;
    LoadStoreQueue lsq;
    ExecutePipe execute;
    LoadStorePipe memory;
    CommitPipe commit;

private:
    std::size_t rename_width_;
};

} // namespace riscv_cpu
