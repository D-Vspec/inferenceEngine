#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <variant>
#include <span>
#include <cstdint>
#include <array>

#include "parser.h"

using TData = std::variant<

    // Activation, KV Cache and all
    std::vector<float>,
    std::vector<uint16_t>,
    std::vector<uint8_t>,
    std::vector<int8_t>,
    std::vector<int16_t>,
    std::vector<int32_t>,
    std::vector<int64_t>,
    std::vector<double>,

    // For weights and stuff
    std::span<const float>,
    std::span<const uint16_t>,
    std::span<const uint8_t>,
    std::span<const int8_t>,
    std::span<const int16_t>,
    std::span<const int32_t>,
    std::span<const int64_t>,
    std::span<const double>
>;

typedef struct {
    std::vector<uint64_t> dims;
    std::vector<uint64_t> stride;
    bool mut;
    TData data; 
} Tensor;

Tensor loadTensor(const char* tensorLocation, bool mut, uint32_t size, uint32_t type);

void addTensors(const Tensor& a, const Tensor& b, Tensor& out); //A + B
void mulTensors(const Tensor& a, const Tensor& b, Tensor& out); //A * B
void subTensors(const Tensor& a, const Tensor& b, Tensor& out); //A - B 
void printTensor(const Tensor& tensor);
Tensor transpose(const Tensor& a);
double sum(const Tensor& a);

#endif