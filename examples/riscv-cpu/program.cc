#include "program.hh"

#include <fstream>
#include <stdexcept>

namespace riscv_cpu {

std::vector<std::uint8_t> loadRawImage(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open raw program: " + path);
    }

    std::vector<std::uint8_t> bytes(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    if (bytes.empty()) {
        throw std::runtime_error("raw program is empty: " + path);
    }
    if (bytes.size() % 4 != 0) {
        throw std::runtime_error("raw program size is not a multiple of 4 bytes");
    }
    return bytes;
}

} // namespace riscv_cpu
