#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstdint>

#include "parser.h"

struct Buffer {
    ggml_type dtype = GGML_TYPE_F32;
    void* rawData = nullptr;
    size_t numElements = 0;
    bool isOwned = false;

    Buffer() = default;

    // No copy — prevent accidental double-free
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    // Move
    Buffer(Buffer&& other) noexcept
        : dtype(other.dtype)
        , rawData(other.rawData)
        , numElements(other.numElements)
        , isOwned(other.isOwned)
    {
        other.rawData = nullptr;
        other.isOwned = false;
    }

    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            dtype = other.dtype;
            rawData = other.rawData;
            numElements = other.numElements;
            isOwned = other.isOwned;
            other.rawData = nullptr;
            other.isOwned = false;
        }
        return *this;
    }
};

typedef struct {
    std::vector<uint64_t> dims;
    std::vector<uint64_t> stride;
    bool mut;
    Buffer data;
} Tensor;

// Buffer helpers
const float* asFloatPtr(const Buffer& buf);
float* asMutableFloatPtr(Buffer& buf);
std::vector<float> dequantizeToFloat(const Buffer& buf);
size_t bufferByteSize(const Buffer& buf);

Tensor loadTensor(const char* tensorLocation, bool mut, uint32_t size, uint32_t type);

void addTensors(const Tensor& a, const Tensor& b, Tensor& out); //A + B
void mulTensors(const Tensor& a, const Tensor& b, Tensor& out); //A * B
void subTensors(const Tensor& a, const Tensor& b, Tensor& out); //A - B 
void printTensor(const Tensor& tensor);
Tensor transpose(const Tensor& a);
double sum(const Tensor& a);

#endif