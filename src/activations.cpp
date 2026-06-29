#include "../headers/activations.h"
#include "../headers/tensor.h"

#include <stdexcept>
#include <cstdlib>
#include <cmath>

// ─── RMS Normalization ────────────────────────────────────────────────────────

Tensor rmsNorm(const Tensor& x, const Tensor& weight, float epsilon) {
    size_t n = x.data.numElements;

    if (weight.data.numElements != n)
        throw std::invalid_argument("rmsNorm: weight must have same number of elements as input");

    const float* xData = asFloatPtr(x.data);
    std::vector<float> xDequant;
    if (!xData) {
        xDequant = dequantizeToFloat(x.data);
        xData = xDequant.data();
    }

    const float* wData = asFloatPtr(weight.data);
    std::vector<float> wDequant;
    if (!wData) {
        wDequant = dequantizeToFloat(weight.data);
        wData = wDequant.data();
    }

    // mean(x^2)
    double sumSq = 0.0;
    for (size_t i = 0; i < n; ++i)
        sumSq += static_cast<double>(xData[i]) * static_cast<double>(xData[i]);
    double meanSq = sumSq / static_cast<double>(n);

    float rsqrt = 1.0f / std::sqrt(static_cast<float>(meanSq) + epsilon);

    float* outData = static_cast<float*>(std::malloc(n * sizeof(float)));
    if (!outData) throw std::bad_alloc();

    for (size_t i = 0; i < n; ++i)
        outData[i] = xData[i] * rsqrt * wData[i];

    Tensor out;
    out.dims = {static_cast<uint64_t>(n)};
    out.stride = {1};
    out.mut = true;
    out.data = Buffer(GGML_TYPE_F32, outData, n, true);
    return out;
}

// ─── ReLU activation ──────────────────────────────────────────────────────────

Tensor relu(const Tensor& x) {
    size_t n = x.data.numElements;

    const float* xData = asFloatPtr(x.data);
    std::vector<float> xDequant;
    if (!xData) {
        xDequant = dequantizeToFloat(x.data);
        xData = xDequant.data();
    }

    float* outData = static_cast<float*>(std::malloc(n * sizeof(float)));
    if (!outData) throw std::bad_alloc();

    for (size_t i = 0; i < n; ++i)
        outData[i] = xData[i] > 0.0f ? xData[i] : 0.0f;

    Tensor out;
    out.dims = {static_cast<uint64_t>(n)};
    out.stride = {1};
    out.mut = true;
    out.data = Buffer(GGML_TYPE_F32, outData, n, true);
    return out;
}

// ─── SiLU / Swish activation ──────────────────────────────────────────────────

Tensor silu(const Tensor& x) {
    size_t n = x.data.numElements;

    const float* xData = asFloatPtr(x.data);
    std::vector<float> xDequant;
    if (!xData) {
        xDequant = dequantizeToFloat(x.data);
        xData = xDequant.data();
    }

    float* outData = static_cast<float*>(std::malloc(n * sizeof(float)));
    if (!outData) throw std::bad_alloc();

    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    for (size_t i = 0; i < n; ++i)
        outData[i] = xData[i] / (1.0f + std::exp(-xData[i]));

    Tensor out;
    out.dims = {static_cast<uint64_t>(n)};
    out.stride = {1};
    out.mut = true;
    out.data = Buffer(GGML_TYPE_F32, outData, n, true);
    return out;
}

// ─── SwiGLU gating ────────────────────────────────────────────────────────────

Tensor swiglu(const Tensor& gate, const Tensor& up) {
    size_t n = gate.data.numElements;

    if (up.data.numElements != n)
        throw std::invalid_argument("swiglu: gate and up must have same number of elements");

    const float* gateData = asFloatPtr(gate.data);
    std::vector<float> gateDequant;
    if (!gateData) {
        gateDequant = dequantizeToFloat(gate.data);
        gateData = gateDequant.data();
    }

    const float* upData = asFloatPtr(up.data);
    std::vector<float> upDequant;
    if (!upData) {
        upDequant = dequantizeToFloat(up.data);
        upData = upDequant.data();
    }

    float* outData = static_cast<float*>(std::malloc(n * sizeof(float)));
    if (!outData) throw std::bad_alloc();

    // swiglu(gate, up) = silu(gate) * up  (element-wise)
    for (size_t i = 0; i < n; ++i)
        outData[i] = (gateData[i] / (1.0f + std::exp(-gateData[i]))) * upData[i];

    Tensor out;
    out.dims = {static_cast<uint64_t>(n)};
    out.stride = {1};
    out.mut = true;
    out.data = Buffer(GGML_TYPE_F32, outData, n, true);
    return out;
}
