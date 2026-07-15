#pragma once

#include "channel.hh"
#include "params.hh"
#include "signal.hh"

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace coropulse {

struct SimulatorConfig {
    std::size_t workers = 1;
    std::size_t load_balance_window = 0;
};

class Simulator {
public:
    explicit Simulator(std::size_t workers = 1) : Simulator(SimulatorConfig{workers, 0}) {}

    explicit Simulator(SimulatorConfig config)
        : runtime_(config.workers) {
        if (config.load_balance_window != 0) {
            runtime_.enableLoadBalancing(config.load_balance_window);
        }
    }

    template <class ComponentT, class... Args>
    ComponentT& createComponent(Args&&... args) {
        static_assert(std::is_base_of_v<Component, ComponentT>,
                      "createComponent<T>() requires T to derive from Component");
        ensureMutableTopology();

        auto component = std::make_unique<ComponentT>(std::forward<Args>(args)...);
        auto& ref = *component;
        runtime_.addComponent(ref);
        components_.push_back(std::move(component));
        return ref;
    }

    template <class T, class... Inputs>
    void connect(Output<T>& output, Input<T>& first_input, Inputs&... rest_inputs) {
        static_assert((std::is_same_v<Input<T>, std::remove_cvref_t<Inputs>> && ...),
                      "connect(output, ...) requires all inputs to have the same value type");
        ensureMutableTopology();

        auto channel = std::make_unique<Channel<T>>(output.name());
        auto& ref = *channel;
        output.bind(ref);
        first_input.bind(ref);
        (rest_inputs.bind(ref), ...);
        runtime_.addObject(ref);
        objects_.push_back(std::move(channel));
    }

    template <class T>
    void connect(Output<T>& output, const std::vector<Input<T>*>& inputs) {
        ensureMutableTopology();
        if (inputs.empty()) {
            throw std::runtime_error("connect(output, inputs) requires at least one input");
        }

        auto channel = std::make_unique<Channel<T>>(output.name());
        auto& ref = *channel;
        output.bind(ref);
        for (auto* input : inputs) {
            if (!input) {
                throw std::runtime_error("connect(output, inputs) received a null input");
            }
            input->bind(ref);
        }
        runtime_.addObject(ref);
        objects_.push_back(std::move(channel));
    }

    template <class T>
    void connect(Output<T>& output, std::initializer_list<Input<T>*> inputs) {
        connect(output, std::vector<Input<T>*>{inputs});
    }

    template <class T, class... Inputs>
    void connect(SignalOutput<T>& output, SignalInput<T>& first_input,
                 Inputs&... rest_inputs) {
        static_assert((std::is_same_v<SignalInput<T>, std::remove_cvref_t<Inputs>> && ...),
                      "connect(signal_output, ...) requires all inputs to have the same value type");
        ensureMutableTopology();

        auto signal = std::make_unique<Signal<T>>(output.name());
        auto& ref = *signal;
        output.bind(ref);
        first_input.bind(ref);
        (rest_inputs.bind(ref), ...);
        runtime_.addObject(ref);
        objects_.push_back(std::move(signal));
    }

    template <class T>
    void connect(SignalOutput<T>& output, const std::vector<SignalInput<T>*>& inputs) {
        ensureMutableTopology();
        if (inputs.empty()) {
            throw std::runtime_error("connect(signal_output, inputs) requires at least one input");
        }

        auto signal = std::make_unique<Signal<T>>(output.name());
        auto& ref = *signal;
        output.bind(ref);
        for (auto* input : inputs) {
            if (!input) {
                throw std::runtime_error(
                    "connect(signal_output, inputs) received a null input");
            }
            input->bind(ref);
        }
        runtime_.addObject(ref);
        objects_.push_back(std::move(signal));
    }

    template <class T>
    void connect(SignalOutput<T>& output, std::initializer_list<SignalInput<T>*> inputs) {
        connect(output, std::vector<SignalInput<T>*>{inputs});
    }

    void enableLoadBalancing(std::size_t window_ticks) {
        runtime_.enableLoadBalancing(window_ticks);
    }

    void disableLoadBalancing() noexcept {
        runtime_.disableLoadBalancing();
    }

    void tick() {
        topology_frozen_ = true;
        runtime_.runTick();
    }

    void run(std::size_t ticks) {
        for (std::size_t i = 0; i < ticks; ++i) {
            tick();
        }
    }

    TickId currentTick() const noexcept {
        return runtime_.tick();
    }

private:
    void ensureMutableTopology() const {
        if (topology_frozen_) {
            throw std::runtime_error(
                "cannot create components or connections after simulation has started");
        }
    }

    std::vector<std::unique_ptr<Component>> components_;
    std::vector<std::unique_ptr<TickObject>> objects_;
    Runtime runtime_;
    bool topology_frozen_ = false;
};

} // namespace coropulse
