#include "backend_pipeline.hh"

#include <stdexcept>
#include <utility>

namespace riscv_cpu {

bool RenamePipe::slotOpen() const noexcept {
    return !reg_;
}

bool RenamePipe::hasIssueBundle() const noexcept {
    return reg_ && !issue_consumed_;
}

const InstBundle& RenamePipe::issueBundle() const {
    if (!reg_ || issue_consumed_) {
        throw std::runtime_error("rename pipe has no issue-visible bundle");
    }
    return *reg_;
}

const InstBundle* RenamePipe::readCommitDispatch() {
    if (!reg_ || commit_consumed_) {
        return nullptr;
    }

    commit_consumed_ = true;
    return &*reg_;
}

void RenamePipe::markIssueConsumed() {
    if (!reg_ || issue_consumed_) {
        throw std::runtime_error("rename pipe issue side consumed an empty bundle");
    }
    issue_consumed_ = true;
}

void RenamePipe::clearIfConsumed() {
    if (reg_ && issue_consumed_ && commit_consumed_) {
        clear();
    }
}

void RenamePipe::accept(CoreState& core, InstBundle&& bundle, BackendStats& stats) {
    if (reg_) {
        throw std::runtime_error("rename pipe register is already occupied");
    }

    for (auto* inst : bundle) {
        core.rename(inst);
    }
    stats.renamed += bundle.size();
    reg_ = std::move(bundle);
    issue_consumed_ = false;
    commit_consumed_ = false;
}

void RenamePipe::clear() noexcept {
    reg_.reset();
    issue_consumed_ = false;
    commit_consumed_ = false;
}

} // namespace riscv_cpu
