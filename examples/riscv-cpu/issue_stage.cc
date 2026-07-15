#include "stages.hh"

namespace riscv_cpu {

IssueStage::IssueStage(std::size_t capacity) : capacity_(capacity) {}

coropulse::Task<void> IssueStage::process() {
    for (;; co_yield coropulse::tickDone{}) {
        if (redirect_in.read()) {
            queue_.clear();
            completed_.clear();
            (void)completion_in.read();
            (void)rename_in.read();
            can_accept.set(true);
            continue;
        }

        if (auto completion = completion_in.read()) {
            rememberCompletion(*completion);
        }

        const auto ready = findReady();
        if (ready) {
            auto& entry = queue_[*ready];
            auto& rename = entry.inst->renameState();
            rename.src1 = entry.src1;
            rename.src2 = entry.src2;

            if (issue_out.write(entry.inst)) {
                queue_.erase(queue_.begin() + static_cast<std::ptrdiff_t>(*ready));
                ++issued_;
            } else {
                ++output_stalls_;
            }
        }

        const bool input_ready = queue_.size() < capacity_;
        can_accept.set(input_ready);

        if (input_ready) {
            if (auto renamed = rename_in.read()) {
                auto entry = makeEntry(*renamed);
                applyKnownWakeups(entry);
                queue_.push_back(entry);
                ++accepted_;
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

std::optional<std::size_t> IssueStage::findReady() const {
    bool older_memory_waiting = false;
    for (std::size_t i = 0; i < queue_.size(); ++i) {
        const auto& entry = queue_[i];
        const bool memory_blocked = entry.memory && older_memory_waiting;
        if (operandsReady(entry) && !memory_blocked) {
            return i;
        }
        if (entry.memory) {
            older_memory_waiting = true;
        }
    }
    return std::nullopt;
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
