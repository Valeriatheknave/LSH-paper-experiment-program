CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -march=native -ffast-math -fopenmp -Wall -Wextra -pedantic
LDFLAGS ?= -fopenmp

SRCS := src/db_lsh.cpp src/main.cpp
OBJ := $(SRCS:.cpp=.o)
TARGET := dblsh2

.PHONY: all clean run test

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.cpp src/db_lsh.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) audio 1.5 50 5 10 0.1 0.3

# Small correctness check: beta=1 must return the exact top-k.
test: $(TARGET)
	./$(TARGET) audio 1.5 10 5 10 1.0 0.3 --queries 5 --out /tmp/dblsh2-test.csv

clean:
	rm -f $(OBJ) $(TARGET)
