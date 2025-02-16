
CXX = g++
# LDFLAGS = -lstdc++ -std=c++17 -O2 -g -fsanitize=address
LDFLAGS = -lstdc++ -std=c++17 -O0 -g -fsanitize=address

CSRC_DIR = ../src/_ngram
INCLUDES = -I$(CSRC_DIR) 

TEST_SOURCES = $(wildcard *_test.cpp)
TEST_EXECUTABLES = $(TEST_SOURCES:.cpp=)

.PHONY: test clean test_cpp test_py prof

compile_test: $(TEST_EXECUTABLES) $(CSRC_DIR)

%: %.cpp
	$(CXX) $(LDFLAGS) $(INCLUDES) -o $@ $<

test_cpp: compile_test
	@for exec in $(TEST_EXECUTABLES); do ./$$exec; done

prof_cpp: compile_test
	@for exec in $(TEST_EXECUTABLES); do \
	 perf stat -e L1-dcache-load-misses,L1-dcache-loads,L1-dcache-stores,L1-icache-load-misses,lock:lock_acquire,lock:lock_release ./$$exec ; done

test_py: 
	pytest . -v

test: clean test_cpp test_py
	make clean

prof: prof_cpp

clean:
	rm -f $(TEST_EXECUTABLES)