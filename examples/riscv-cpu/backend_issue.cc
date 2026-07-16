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

void IssuePipe::rememberCompletions(const ExecResultBundle& completions) {
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

void IssuePipe::issue(coropulse::Output<InstBundle>& output, BackendStats& stats) {
    const auto ready = findReadyBundle();
    if (ready.empty()) {
        return;
    }

    InstBundle bundle;
    bundle.reserve(ready.size());
    for (auto index : ready) {
        auto& entry = queue_[index];
        auto& rename = entry.inst->renameState();
        rename.src1 = entry.src1;
        rename.src2 = entry.src2;
        bundle.push_back(entry.inst);
    }

    if (output.write(std::move(bundle))) {
        for (auto iter = ready.rbegin(); iter != ready.rend(); ++iter) {
            queue_.erase(queue_.begin() + static_cast<std::ptrdiff_t>(*iter));
        }
        stats.issued += ready.size();
    } else {
        ++stats.output_stalls;
    }
}

void IssuePipe::clear() {
    queue_.clear();
    completed_.clear();
}

bool IssuePipe::operandsReady(const Entry& entry) const {
    return entry.src1.ready && entry.src2.ready;
}

bool IssuePipe::canIssue(const Entry& entry) const {
    if (!operandsReady(entry)) {
        return false;
    }
    if (!entry.memory) {
        return true;
    }
    return core_.memoryOrderReady(entry.inst->renameState().sequence);
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
    const auto& rename = inst->renameState();
    return Entry{
        inst,
        rename.src1,
        rename.src2,
        rename.memory,
    };
}

void IssuePipe::rememberCompletion(const ExecResult& completion) {
    if (completion.sequence >= completed_.size()) {
        completed_.resize(completion.sequence + 1);
    }
    completed_[completion.sequence] = completion;
    applyWakeup(completion);
}

void IssuePipe::applyWakeup(const ExecResult& completion) {
    if (!completion.writes_rd) {
        return;
    }

    for (auto& entry : queue_) {
        wakeOperand(entry.src1, completion);
        wakeOperand(entry.src2, completion);
    }
}

void IssuePipe::applyKnownWakeups(Entry& entry) const {
    applyKnownWakeup(entry.src1);
    applyKnownWakeup(entry.src2);
}

void IssuePipe::applyKnownWakeup(Operand& operand) const {
    if (operand.ready || operand.producer >= completed_.size()) {
        return;
    }

    const auto& completion = completed_[operand.producer];
    if (completion && completion->writes_rd) {
        operand.ready = true;
        operand.value = completion->value;
    }
}

void IssuePipe::wakeOperand(Operand& operand, const ExecResult& completion) {
    if (!operand.ready && operand.producer == completion.sequence) {
        operand.ready = true;
        operand.value = completion.value;
    }
}

} // namespace riscv_cpu
