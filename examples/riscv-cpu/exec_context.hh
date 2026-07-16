#pragma once

#include "pipeline_types.hh"

#include <cstddef>
#include <cstdint>

namespace riscv_cpu {

class CoreState;
class DynInst;
class SimpleSram;

class ExecContext {
public:
    ExecContext(CoreState& core, SimpleSram& sram, DynInst& inst,
                coropulse::TickId tick);

    const StaticInst& staticInst() const noexcept;
    std::uint64_t pc() const noexcept;
    std::uint64_t predictedNextPc() const noexcept;
    const BranchPrediction& branchPrediction() const noexcept;

    std::uint64_t readSrc1() const noexcept;
    std::uint64_t readSrc2() const noexcept;
    void writeResult(std::uint64_t value) noexcept;

    std::uint64_t load(std::uint64_t address, std::size_t bytes,
                       bool sign_extend) const;
    void setStore(std::uint64_t address, std::uint64_t value,
                  std::size_t bytes);
    void setRedirect(ControlRedirect redirect);
    void recordBranch(bool conditional, bool taken, std::uint64_t target,
                      std::uint64_t actual_next_pc);
    void suppressPredictedRedirect();

private:
    CoreState& core_;
    SimpleSram& sram_;
    DynInst& inst_;
};

} // namespace riscv_cpu
