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
EXAMPLE_BINS := $(patsubst examples/%.cpp,$(BUILD_DIR)/%,$(EXAMPLE_SRCS))

.PHONY: all test examples issue-queue-example rate-test signal-rate-test clean

all: test

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TEST_BINS): $(BUILD_DIR)/%: tests/%.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(THREAD_FLAGS) $< -o $@

$(BENCH_BINS): $(BUILD_DIR)/%: benchmarks/%.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(THREAD_FLAGS) $< -o $@

$(EXAMPLE_BINS): $(BUILD_DIR)/%: examples/%.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(THREAD_FLAGS) $< -o $@

test: $(TEST_BINS)
	@for test in $(TEST_BINS); do ./$$test; done

examples: $(EXAMPLE_BINS)
	@for example in $(EXAMPLE_BINS); do ./$$example; done

issue-queue-example: $(BUILD_DIR)/issue_queue_wakeup
	./$(BUILD_DIR)/issue_queue_wakeup

rate-test: $(BUILD_DIR)/rate_bench $(BUILD_DIR)/signal_rate_bench
	./$(BUILD_DIR)/rate_bench
	./$(BUILD_DIR)/signal_rate_bench

signal-rate-test: $(BUILD_DIR)/signal_rate_bench
	./$(BUILD_DIR)/signal_rate_bench

clean:
	rm -rf $(BUILD_DIR)
