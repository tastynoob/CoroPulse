#include "backend_pipeline.hh"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace riscv_cpu {

IssuePipe::IssuePipe(CoreState& core, std::size_t capacity,
                     std::size_t dispatch_width, std::size_t issue_width)
    : core_(core),
      capacity_(capacity),
      dispatch_width_(dispatch_width),
      issue_width_(issue_width) {
    if (capacity_ == 0 || dispatch_width_ == 0 || issue_width_ == 0) {
        throw std::runtime_error("issue capacity and widths must be greater than zero");
    }
    if (capacity_ < dispatch_width_) {
        throw std::runtime_error("issue capacity must cover dispatch width");
    }
}

bool IssuePipe::canAcceptDispatch() const {
    return queue_.size() + dispatch_width_ <= capacity_;
}

void IssuePipe::rememberCompletions(const InstBundle& completions) {
    for (const auto& completion : completions) {
        rememberCompletion(completion);
    }
}

void IssuePipe::acceptRenamed(const InstBundle& bundle, BackendStats& stats) {
    if (!canAcceptDispatch()) {
        throw std::runtime_error("issue pipe accepted a bundle while full");
    }

    for (auto* renamed : bundle) {
        auto entry = makeEntry(renamed);
        applyKnownWakeups(entry);
        queue_.push_back(entry);
    }
    stats.accepted += bundle.size();
}

InstBundle IssuePipe::issue(BackendStats& stats) {
    const auto ready = findReadyBundle();
    if (ready.empty()) {
        return {};
    }

    InstBundle bundle;
    bundle.reserve(ready.size());
    for (auto index : ready) {
        bundle.push_back(queue_[index].inst);
    }

    for (auto iter = ready.rbegin(); iter != ready.rend(); ++iter) {
        queue_.erase(queue_.begin() + static_cast<std::ptrdiff_t>(*iter));
    }
    stats.issued += ready.size();
    return bundle;
}

void IssuePipe::clear() {
    queue_.clear();
    completed_.clear();
}

bool IssuePipe::operandsReady(const Entry& entry) const {
    const auto& rename = entry.inst->renameState();
    return rename.src1.ready && rename.src2.ready;
}

bool IssuePipe::canIssue(const Entry& entry) const {
    if (!operandsReady(entry)) {
        return false;
    }
    const auto& rename = entry.inst->renameState();
    if (!rename.memory) {
        return true;
    }
    return core_.memoryOrderReady(rename.sequence);
}

std::vector<std::size_t> IssuePipe::findReadyBundle() const {
    std::vector<std::size_t> ready;
    for (std::size_t i = 0; i < queue_.size(); ++i) {
        const auto& entry = queue_[i];
        if (canIssue(entry)) {
            ready.push_back(i);
            if (ready.size() >= issue_width_) {
                break;
            }
        }
    }
    return ready;
}

IssuePipe::Entry IssuePipe::makeEntry(DynInstPtr inst) const {
    return Entry{inst};
}

void IssuePipe::rememberCompletion(DynInstPtr completion) {
    if (!completion) {
        throw std::runtime_error("issue pipe received an empty completion");
    }

    const auto sequence = completion->renameState().sequence;
    if (sequence >= completed_.size()) {
        completed_.resize(sequence + 1);
    }
    completed_[sequence] = 1;
    applyWakeup(completion);
}

void IssuePipe::applyWakeup(DynInstPtr completion) {
    const auto& rename = completion->renameState();
    if (!rename.writes_rd) {
        return;
    }

    for (auto& entry : queue_) {
        auto& rename = entry.inst->renameState();
        wakeOperand(rename.src1, completion);
        wakeOperand(rename.src2, completion);
    }
}

void IssuePipe::applyKnownWakeups(Entry& entry) const {
    auto& rename = entry.inst->renameState();
    applyKnownWakeup(rename.src1);
    applyKnownWakeup(rename.src2);
}

void IssuePipe::applyKnownWakeup(Operand& operand) const {
    if (operand.ready || operand.producer >= completed_.size()) {
        return;
    }

    if (completed_[operand.producer] != 0) {
        operand.ready = true;
    }
}

void IssuePipe::wakeOperand(Operand& operand, DynInstPtr producer) {
    const auto& rename = producer->renameState();
    if (!operand.ready && operand.producer == rename.sequence) {
        operand.ready = true;
    }
}

} // namespace riscv_cpu
