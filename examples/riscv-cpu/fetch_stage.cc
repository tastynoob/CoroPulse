#include "stages.hh"

#include <algorithm>
#include <stdexcept>

namespace riscv_cpu {

FetchStage::FetchStage(const SimpleSram& sram, DynInstPool& inst_pool,
                       std::size_t fetch_width)
    : sram_(sram),
      inst_pool_(inst_pool),
      fetch_width_(fetch_width) {
    if (fetch_width_ == 0) {
        throw std::runtime_error("fetch width must be greater than zero");
    }
}

coropulse::Task<void> FetchStage::process() {
    for (;; co_yield coropulse::tickDone{}) {
        bool fifo_ready = true;
        if (auto redirect = redirect_in.read()) {
            applyRedirect(*redirect);
        } else {
            fifo_ready = co_await fifo_can_accept.read();
        }

        if (halted_) {
            continue;
        }
        if (!fifo_ready || !out.canWrite()) {
            if (pc_ / 4 < sram_.instructionCount()) {
                ++stats.backpressure_stalls;
            }
            continue;
        }

        InstBundle bundle;
        bundle.reserve(fetch_width_);
        while (bundle.size() < fetch_width_ && pc_ / 4 < sram_.instructionCount()) {
            const auto fetch_pc = pc_;
            const auto predicted_next_pc = fetch_pc + 4;
            bundle.push_back(inst_pool_.create(sram_.loadInstruction(fetch_pc), fetch_pc,
                                               predicted_next_pc));
            pc_ = predicted_next_pc;
            ++stats.fetched;
        }

        if (!bundle.empty()) {
            (void)out.write(std::move(bundle));
        } else {
            halted_ = true;
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

void FetchStage::applyRedirect(const ControlRedirect& redirect) {
    ++stats.redirects;
    pc_ = redirect.next_pc;
    halted_ = redirect.halt;
    architectural_halted_ = redirect.halt;
}

FetchDecodeFifo::FetchDecodeFifo(std::size_t capacity, std::size_t fetch_width,
                                 std::size_t decode_width)
    : capacity_(capacity),
      fetch_width_(fetch_width),
      decode_width_(decode_width) {
    if (capacity_ == 0) {
        throw std::runtime_error("fetch-decode fifo capacity must be greater than zero");
    }
    if (fetch_width_ == 0 || decode_width_ == 0) {
        throw std::runtime_error("fetch/decode widths must be greater than zero");
    }
    if (capacity_ < fetch_width_) {
        throw std::runtime_error("fetch-decode fifo capacity must cover fetch width");
    }
}

coropulse::Task<void> FetchDecodeFifo::process() {
    for (;; co_yield coropulse::tickDone{}) {
        if (redirect_in.read()) {
            queue_.clear();
            (void)in.take();
            continue;
        }

        const bool backend_ready = co_await backend_can_accept.read();

        if (backend_ready && !queue_.empty() && out.canWrite()) {
            (void)out.write(popDecodeBundle());
        }

        const bool input_ready = canAcceptFetch();
        can_accept.set(input_ready);
        if (!input_ready && in.hasValue()) {
            ++stats.overflow_stalls;
        }

        if (input_ready) {
            if (auto fetched = in.take()) {
                for (auto* inst : *fetched) {
                    queue_.push_back(inst);
                }
                if (queue_.size() > stats.max_occupancy) {
                    stats.max_occupancy = queue_.size();
                }
            }
        }
    }
    co_return;
}

bool FetchDecodeFifo::canAcceptFetch() const {
    return capacity_ >= queue_.size() + fetch_width_;
}

InstBundle FetchDecodeFifo::popDecodeBundle() {
    InstBundle bundle;
    const auto count = std::min(decode_width_, queue_.size());
    bundle.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        bundle.push_back(queue_.front());
        queue_.pop_front();
    }
    return bundle;
}

} // namespace riscv_cpu
