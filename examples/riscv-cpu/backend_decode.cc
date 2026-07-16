#include "backend_pipeline.hh"

#include <stdexcept>
#include <utility>

namespace riscv_cpu {

bool DecodePipe::slotOpen() const noexcept {
    return !reg_;
}

bool DecodePipe::hasBundle() const noexcept {
    return reg_.has_value();
}

InstBundle DecodePipe::takeBundle() {
    if (!reg_) {
        throw std::runtime_error("decode pipe has no bundle to take");
    }

    auto bundle = std::move(*reg_);
    reg_.reset();
    return bundle;
}

void DecodePipe::accept(InstBundle&& bundle, BackendStats& stats) {
    if (reg_) {
        throw std::runtime_error("decode pipe register is already occupied");
    }

    stats.decoded += bundle.size();
    reg_ = std::move(bundle);
}

void DecodePipe::clear() noexcept {
    reg_.reset();
}

} // namespace riscv_cpu
