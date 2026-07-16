#include "stages.hh"

#include <stdexcept>

namespace riscv_cpu {

IssueStage::IssueStage(CoreState& core, std::size_t capacity, std::size_t dispatch_width,
                       std::size_t issue_width)
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

coropulse::Task<void> IssueStage::process() {
    for (;; co_yield coropulse::tickDone{}) {
        if (redirect_in.read()) {
            queue_.clear();
            completed_.clear();
            (void)completion_in.read();
            (void)rename_in.read();
            continue;
        }

        if (auto completions = completion_in.read()) {
            rememberCompletions(*completions);
        }

        const auto ready = findReadyBundle();
        if (!ready.empty()) {
            InstBundle bundle;
            bundle.reserve(ready.size());
            for (auto index : ready) {
                auto& entry = queue_[index];
                auto& rename = entry.inst->renameState();
                rename.src1 = entry.src1;
                rename.src2 = entry.src2;
                bundle.push_back(entry.inst);
            }

            if (issue_out.write(std::move(bundle))) {
                for (auto iter = ready.rbegin(); iter != ready.rend(); ++iter) {
                    queue_.erase(queue_.begin() + static_cast<std::ptrdiff_t>(*iter));
                }
                issued_ += ready.size();
            } else {
                ++output_stalls_;
            }
        }

        const bool input_ready = queue_.size() + dispatch_width_ <= capacity_;
        can_accept.set(input_ready);

        if (input_ready) {
            if (auto renamed_bundle = rename_in.read()) {
                for (auto* renamed : *renamed_bundle) {
                    auto entry = makeEntry(renamed);
                    applyKnownWakeups(entry);
                    queue_.push_back(entry);
                }
                accepted_ += renamed_bundle->size();
            }
        }
    }
    co_return;
}

std::size_t IssueStage::issuedCount() const {
    return issued_;
}

std::size_t IssueStage::acceptedCount() const {
    return accepted_;
}

std::size_t IssueStage::outputStalls() const {
    return output_stalls_;
}

bool IssueStage::operandsReady(const IssueEntry& entry) const {
    return entry.src1.ready && entry.src2.ready;
}

bool IssueStage::canIssue(const IssueEntry& entry) const {
    if (!operandsReady(entry)) {
        return false;
    }
    if (!entry.memory) {
        return true;
    }
    return core_.memoryOrderReady(entry.inst->renameState().sequence);
}

std::vector<std::size_t> IssueStage::findReadyBundle() const {
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

IssueStage::IssueEntry IssueStage::makeEntry(DynInstPtr inst) const {
    const auto& rename = inst->renameState();
    return IssueEntry{
        inst,
        rename.src1,
        rename.src2,
        rename.memory,
    };
}

void IssueStage::rememberCompletion(const ExecResult& completion) {
    if (completion.sequence >= completed_.size()) {
        completed_.resize(completion.sequence + 1);
    }
    completed_[completion.sequence] = completion;
    applyWakeup(completion);
}

void IssueStage::rememberCompletions(const ExecResultBundle& completions) {
    for (const auto& completion : completions) {
        rememberCompletion(completion);
    }
}

void IssueStage::applyWakeup(const ExecResult& completion) {
    if (!completion.writes_rd) {
        return;
    }

    for (auto& entry : queue_) {
        wakeOperand(entry.src1, completion);
        wakeOperand(entry.src2, completion);
    }
}

void IssueStage::applyKnownWakeups(IssueEntry& entry) const {
    applyKnownWakeup(entry.src1);
    applyKnownWakeup(entry.src2);
}

void IssueStage::applyKnownWakeup(Operand& operand) const {
    if (operand.ready || operand.producer >= completed_.size()) {
        return;
    }

    const auto& completion = completed_[operand.producer];
    if (completion && completion->writes_rd) {
        operand.ready = true;
        operand.value = completion->value;
    }
}

void IssueStage::wakeOperand(Operand& operand, const ExecResult& completion) {
    if (!operand.ready && operand.producer == completion.sequence) {
        operand.ready = true;
        operand.value = completion.value;
    }
}

} // namespace riscv_cpu
