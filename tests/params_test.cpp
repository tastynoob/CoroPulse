#include "sim.hh"

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

using namespace coropulse;

namespace {

class ParametrizedComponent final : public Component {
public:
    ParametrizedComponent(const Params& params, std::vector<int>& observed)
        : issue_width_(params["pipeline"]["issue_width"].as<int>()),
          latency_(params["pipeline"]["latency"].as<int>()),
          enabled_(params["pipeline"]["enabled"].as<bool>()),
          name_(params["name"].as<std::string>()),
          observed_(observed) {}

    MAKE_PROCESS({
        if (enabled_) {
            observed_.push_back(issue_width_ + latency_ +
                                static_cast<int>(name_.size()));
        }
    })

private:
    int issue_width_;
    int latency_;
    bool enabled_;
    std::string name_;
    std::vector<int>& observed_;
};

void params_support_nested_object_initialization() {
    Params params = {
        {"name", "dispatch"},
        {"pipeline", {
            {"issue_width", 4},
            {"latency", 2},
            {"enabled", true},
        }},
        {"ratios", Params::array({1, 2.5, 3})},
    };

    assert(params["name"].as<std::string>() == "dispatch");
    assert(params["pipeline"]["issue_width"].as<int>() == 4);
    assert(params["pipeline"]["latency"].as<int>() == 2);
    assert(params["pipeline"]["enabled"].as<bool>());
    assert(params["ratios"][0].as<int>() == 1);
    assert(params["ratios"][1].as<double>() == 2.5);
}

void params_support_mutating_existing_entries() {
    Params params = {
        {"name", "dispatch"},
        {"pipeline", {
            {"issue_width", 4},
            {"enabled", true},
        }},
    };

    params["pipeline"]["issue_width"] = 6;
    params["name"] = "rename";
    assert(params["pipeline"]["issue_width"].as<int>() == 6);
    assert(params["pipeline"]["enabled"].as<bool>());
    assert(params["name"].as<std::string>() == "rename");
}

void params_report_missing_keys_and_type_errors() {
    Params params = {
        {"pipeline", {
            {"issue_width", 4},
        }},
        {"negative", -1},
    };
    const auto& const_params = params;

    bool missing_threw = false;
    try {
        (void)params["pipeline"]["missing"];
    } catch (const std::runtime_error&) {
        missing_threw = true;
    }
    assert(missing_threw);

    bool const_missing_threw = false;
    try {
        (void)const_params["pipeline"]["missing"];
    } catch (const std::runtime_error&) {
        const_missing_threw = true;
    }
    assert(const_missing_threw);

    bool type_threw = false;
    try {
        (void)params["pipeline"]["issue_width"].as<std::string>();
    } catch (const std::runtime_error&) {
        type_threw = true;
    }
    assert(type_threw);

    bool numeric_type_threw = false;
    try {
        (void)params["pipeline"]["issue_width"].as<double>();
    } catch (const std::runtime_error&) {
        numeric_type_threw = true;
    }
    assert(numeric_type_threw);

    bool range_threw = false;
    try {
        (void)params["negative"].as<std::size_t>();
    } catch (const std::runtime_error&) {
        range_threw = true;
    }
    assert(range_threw);
}

void simulator_passes_params_to_component_constructors() {
    Params params = {
        {"name", "dispatch"},
        {"pipeline", {
            {"issue_width", 4},
            {"latency", 2},
            {"enabled", true},
        }},
    };

    Simulator sim;
    std::vector<int> observed;
    sim.createComponent<ParametrizedComponent>(params, observed);

    sim.tick();

    const std::vector<int> expected = {14};
    assert(observed == expected);
}

} // namespace

int main() {
    params_support_nested_object_initialization();
    params_support_mutating_existing_entries();
    params_report_missing_keys_and_type_errors();
    simulator_passes_params_to_component_constructors();
}
