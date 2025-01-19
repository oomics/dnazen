
CXX = g++
# LDFLAGS = -lstdc++ -std=c++17 -O2 -g -fsanitize=address
LDFLAGS = -lstdc++ -std=c++17 -O2

CSRC_DIR = ../src/_ngram
PYBIND_DIR = ../.venv/lib/python3.11/site-packages
INCLUDES = -I$(CSRC_DIR) -I$(PYBIND_DIR)

TEST_SOURCES = $(wildcard *_test.cpp)
TEST_EXECUTABLES = $(TEST_SOURCES:.cpp=)

compile_test: $(TEST_EXECUTABLES)

%: %.cpp
	$(CXX) $(LDFLAGS) $(INCLUDES) -o $@ $<

test: compile_test
	@for exec in $(TEST_EXECUTABLES); do ./$$exec; done

clean:
	rm -f $(TEST_EXECUTABLES)