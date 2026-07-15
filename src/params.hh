#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace coropulse {

class Param;
struct ParamEntry;

class Param {
public:
    using object_t = std::map<std::string, Param, std::less<>>;
    using array_t = std::vector<Param>;

    Param() = default;
    Param(std::nullptr_t) noexcept {}
    Param(bool value) : value_(value) {}
    Param(const char* value) : value_(std::string(value)) {}
    Param(std::string value) : value_(std::move(value)) {}
    Param(std::string_view value) : value_(std::string(value)) {}
    Param(const object_t& value) : value_(value) {}
    Param(object_t&& value) : value_(std::move(value)) {}
    Param(const array_t& value) : value_(value) {}
    Param(array_t&& value) : value_(std::move(value)) {}
    Param(std::initializer_list<ParamEntry> entries);

    template <class T,
              std::enable_if_t<std::is_integral_v<std::remove_cv_t<T>> &&
                                   !std::is_same_v<std::remove_cv_t<T>, bool>,
                               int> = 0>
    Param(T value) : value_(normalizeInteger(value)) {}

    template <class T,
              std::enable_if_t<std::is_floating_point_v<std::remove_cv_t<T>>, int> = 0>
    Param(T value) : value_(static_cast<double>(value)) {}

    static Param object(std::initializer_list<ParamEntry> entries) {
        return Param(entries);
    }

    static Param array(std::initializer_list<Param> values) {
        return Param(array_t(values));
    }

    bool isNull() const noexcept { return std::holds_alternative<std::nullptr_t>(value_); }
    bool isBool() const noexcept { return std::holds_alternative<bool>(value_); }
    bool isInteger() const noexcept { return std::holds_alternative<std::int64_t>(value_); }
    bool isFloat() const noexcept { return std::holds_alternative<double>(value_); }
    bool isString() const noexcept { return std::holds_alternative<std::string>(value_); }
    bool isObject() const noexcept { return std::holds_alternative<object_t>(value_); }
    bool isArray() const noexcept { return std::holds_alternative<array_t>(value_); }

    const char* typeName() const noexcept {
        switch (value_.index()) {
        case 0:
            return "null";
        case 1:
            return "bool";
        case 2:
            return "integer";
        case 3:
            return "float";
        case 4:
            return "string";
        case 5:
            return "object";
        case 6:
            return "array";
        default:
            return "unknown";
        }
    }

    bool contains(std::string_view key) const {
        const auto* object = std::get_if<object_t>(&value_);
        return object && object->find(key) != object->end();
    }

    Param* find(std::string_view key) {
        auto* object = std::get_if<object_t>(&value_);
        if (!object) {
            return nullptr;
        }

        const auto it = object->find(key);
        if (it == object->end()) {
            return nullptr;
        }
        return &it->second;
    }

    const Param* find(std::string_view key) const {
        const auto* object = std::get_if<object_t>(&value_);
        if (!object) {
            return nullptr;
        }

        const auto it = object->find(key);
        if (it == object->end()) {
            return nullptr;
        }
        return &it->second;
    }

    Param& operator[](std::string_view key) {
        auto* object = std::get_if<object_t>(&value_);
        if (!object) {
            throw std::runtime_error("parameter is not an object; got " +
                                     std::string(typeName()));
        }

        const auto it = object->find(key);
        if (it == object->end()) {
            throw std::runtime_error("missing parameter key: " + std::string(key));
        }
        return it->second;
    }

    const Param& operator[](std::string_view key) const {
        const auto* value = find(key);
        if (!value) {
            throw std::runtime_error("missing parameter key: " + std::string(key));
        }
        return *value;
    }

    Param& operator[](std::size_t index) {
        auto* array = std::get_if<array_t>(&value_);
        if (!array) {
            throw std::runtime_error("parameter is not an array; got " +
                                     std::string(typeName()));
        }
        if (index >= array->size()) {
            throw std::runtime_error("parameter array index out of range");
        }
        return (*array)[index];
    }

    const Param& operator[](std::size_t index) const {
        const auto* array = std::get_if<array_t>(&value_);
        if (!array) {
            throw std::runtime_error("parameter is not an array; got " +
                                     std::string(typeName()));
        }
        if (index >= array->size()) {
            throw std::runtime_error("parameter array index out of range");
        }
        return (*array)[index];
    }

    template <class T>
    T as() const {
        using Result = std::remove_cv_t<std::remove_reference_t<T>>;

        if constexpr (std::is_same_v<Result, Param>) {
            return *this;
        } else if constexpr (std::is_same_v<Result, bool>) {
            const auto* value = std::get_if<bool>(&value_);
            if (!value) {
                throwTypeError("bool");
            }
            return *value;
        } else if constexpr (std::is_integral_v<Result>) {
            const auto* value = std::get_if<std::int64_t>(&value_);
            if (!value) {
                throwTypeError("integer");
            }
            if constexpr (std::is_unsigned_v<Result>) {
                if (*value < 0 ||
                    static_cast<std::uint64_t>(*value) >
                        static_cast<std::uint64_t>(std::numeric_limits<Result>::max())) {
                    throw std::runtime_error("parameter integer is out of requested range");
                }
            } else {
                if (*value < static_cast<std::int64_t>(std::numeric_limits<Result>::min()) ||
                    *value > static_cast<std::int64_t>(std::numeric_limits<Result>::max())) {
                    throw std::runtime_error("parameter integer is out of requested range");
                }
            }
            return static_cast<Result>(*value);
        } else if constexpr (std::is_floating_point_v<Result>) {
            if (const auto* value = std::get_if<double>(&value_)) {
                return static_cast<Result>(*value);
            }
            throwTypeError("float");
        } else if constexpr (std::is_same_v<Result, std::string>) {
            const auto* value = std::get_if<std::string>(&value_);
            if (!value) {
                throwTypeError("string");
            }
            return *value;
        } else if constexpr (std::is_same_v<Result, object_t>) {
            const auto* value = std::get_if<object_t>(&value_);
            if (!value) {
                throwTypeError("object");
            }
            return *value;
        } else if constexpr (std::is_same_v<Result, array_t>) {
            const auto* value = std::get_if<array_t>(&value_);
            if (!value) {
                throwTypeError("array");
            }
            return *value;
        } else {
            static_assert(dependentFalse<Result>(), "unsupported Param::as<T>() type");
        }
    }

private:
    template <class>
    static constexpr bool dependentFalse() {
        return false;
    }

    [[noreturn]] void throwTypeError(const char* expected) const {
        throw std::runtime_error("parameter type mismatch: expected " +
                                 std::string(expected) + ", got " +
                                 std::string(typeName()));
    }

    template <class T>
    static std::int64_t normalizeInteger(T value) {
        using Source = std::remove_cv_t<T>;
        if constexpr (std::is_unsigned_v<Source>) {
            if (value > static_cast<std::uint64_t>(
                            std::numeric_limits<std::int64_t>::max())) {
                throw std::runtime_error("parameter integer is out of storage range");
            }
        }
        return static_cast<std::int64_t>(value);
    }

    std::variant<std::nullptr_t, bool, std::int64_t, double, std::string,
                 object_t, array_t>
        value_{nullptr};
};

struct ParamEntry {
    std::string key;
    Param value;

    ParamEntry(std::string key_value, Param param_value)
        : key(std::move(key_value)), value(std::move(param_value)) {}
};

inline Param::Param(std::initializer_list<ParamEntry> entries)
    : value_(object_t{}) {
    auto& object = std::get<object_t>(value_);
    for (const auto& entry : entries) {
        object[entry.key] = entry.value;
    }
}

using Params = Param;

} // namespace coropulse
