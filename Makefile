CXX ?= g++
CXXFLAGS ?= -std=c++20 -Wall -Wextra -Wpedantic -O2 -g
CPPFLAGS ?= -Iinclude
THREAD_FLAGS ?= -pthread

BUILD_DIR := build
HEADERS := $(wildcard include/cpas/*.hpp)
TEST_SRCS := $(wildcard tests/*_test.cpp)
TEST_BINS := $(patsubst tests/%.cpp,$(BUILD_DIR)/%,$(TEST_SRCS))
BENCH_SRCS := $(wildcard benchmarks/*_bench.cpp)
BENCH_BINS := $(patsubst benchmarks/%.cpp,$(BUILD_DIR)/%,$(BENCH_SRCS))

.PHONY: all test rate-test signal-rate-test clean

all: test

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TEST_BINS): $(BUILD_DIR)/%: tests/%.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(THREAD_FLAGS) $< -o $@

$(BENCH_BINS): $(BUILD_DIR)/%: benchmarks/%.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(THREAD_FLAGS) $< -o $@

test: $(TEST_BINS)
	@for test in $(TEST_BINS); do ./$$test; done

rate-test: $(BUILD_DIR)/rate_bench $(BUILD_DIR)/signal_rate_bench
	./$(BUILD_DIR)/rate_bench
	./$(BUILD_DIR)/signal_rate_bench

signal-rate-test: $(BUILD_DIR)/signal_rate_bench
	./$(BUILD_DIR)/signal_rate_bench

clean:
	rm -rf $(BUILD_DIR)
