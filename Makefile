CXX = g++
CXXFLAGS = -std=c++23 -Wall -Wextra -O2

SRC = main.cpp src/loader.cpp src/parser.cpp src/engine.cpp src/tokenizer.cpp src/tensor.cpp src/util.cpp
BUILD_DIR = build
OBJ = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SRC))
TARGET = inference

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean
