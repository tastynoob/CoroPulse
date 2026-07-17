#include "backend_pipeline.hh"

#include <stdexcept>
#include <utility>

namespace riscv_cpu {

bool RenamePipe::slotOpen() const noexcept {
    return !reg_;
}

bool RenamePipe::hasBundle() const noexcept {
    return reg_.has_value();
}

const InstBundle& RenamePipe::bundle() const {
    if (!reg_) {
        throw std::runtime_error("rename pipe has no bundle");
    }
    return *reg_;
}

InstBundle RenamePipe::takeBundle() {
    if (!reg_) {
        throw std::runtime_error("rename pipe has no bundle to take");
    }

    auto bundle = std::move(*reg_);
    reg_.reset();
    return bundle;
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
}

void RenamePipe::clear() noexcept {
    reg_.reset();
}

} // namespace riscv_cpu
