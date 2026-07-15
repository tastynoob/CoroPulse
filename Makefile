CXX ?= g++
CXXFLAGS ?= -std=c++20 -Wall -Wextra -Wpedantic -O2 -g
CPPFLAGS ?= -Isrc
THREAD_FLAGS ?= -pthread

BUILD_DIR := build
HEADERS := $(wildcard src/*.hh)
TEST_SRCS := $(wildcard tests/*_test.cpp)
TEST_BINS := $(patsubst tests/%.cpp,$(BUILD_DIR)/%,$(TEST_SRCS))
BENCH_SRCS := $(wildcard benchmarks/*_bench.cpp)
BENCH_BINS := $(patsubst benchmarks/%.cpp,$(BUILD_DIR)/%,$(BENCH_SRCS))
EXAMPLE_SRCS := $(wildcard examples/*.cpp)
SIMPLE_EXAMPLE_BINS := $(patsubst examples/%.cpp,$(BUILD_DIR)/%,$(EXAMPLE_SRCS))
RISCV_CPU_SRCS := $(wildcard examples/riscv-cpu/*.cc)
RISCV_CPU_HEADERS := $(wildcard examples/riscv-cpu/*.hh)
RISCV_CPU_BIN := $(BUILD_DIR)/riscv_cpu
RISCV_CPU_RAW ?=

.PHONY: all test examples issue-queue-example riscv-cpu-example rate-test signal-rate-test clean

all: test

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TEST_BINS): $(BUILD_DIR)/%: tests/%.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(THREAD_FLAGS) $< -o $@

$(BENCH_BINS): $(BUILD_DIR)/%: benchmarks/%.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(THREAD_FLAGS) $< -o $@

$(SIMPLE_EXAMPLE_BINS): $(BUILD_DIR)/%: examples/%.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(THREAD_FLAGS) $< -o $@

$(RISCV_CPU_BIN): $(RISCV_CPU_SRCS) $(RISCV_CPU_HEADERS) $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(THREAD_FLAGS) $(RISCV_CPU_SRCS) -o $@

test: $(TEST_BINS)
	@for test in $(TEST_BINS); do ./$$test; done

examples: $(SIMPLE_EXAMPLE_BINS) $(RISCV_CPU_BIN)
	@for example in $(SIMPLE_EXAMPLE_BINS); do ./$$example; done

issue-queue-example: $(BUILD_DIR)/issue_queue_wakeup
	./$(BUILD_DIR)/issue_queue_wakeup

riscv-cpu-example: $(RISCV_CPU_BIN)
	@test -n "$(RISCV_CPU_RAW)" || \
		(echo "set RISCV_CPU_RAW=/path/to/program.bin" >&2; exit 1)
	./$(RISCV_CPU_BIN) $(RISCV_CPU_RAW)

rate-test: $(BUILD_DIR)/rate_bench $(BUILD_DIR)/signal_rate_bench
	./$(BUILD_DIR)/rate_bench
	./$(BUILD_DIR)/signal_rate_bench

signal-rate-test: $(BUILD_DIR)/signal_rate_bench
	./$(BUILD_DIR)/signal_rate_bench

clean:
	rm -rf $(BUILD_DIR)
