CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O2

SRC = main.cpp src/loader.cpp
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
