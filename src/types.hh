#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace coropulse {

using TickId = std::uint64_t;

inline std::string objectError(std::string_view kind, std::string_view name,
                               std::string_view message) {
    std::string out;
    out.reserve(kind.size() + name.size() + message.size() + 6);
    out.append(kind);
    out.append(" '");
    out.append(name);
    out.append("': ");
    out.append(message);
    return out;
}

} // namespace coropulse
