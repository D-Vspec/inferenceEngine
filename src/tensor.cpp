#include "../headers/tensor.h"
#include "../headers/parser.h"
#include "../headers/util.h"

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <thread>
#include <algorithm>

// ─── Buffer helpers ───────────────────────────────────────────────────────────

const float* asFloatPtr(const Buffer& buf) {
    if (buf.dtype == GGML_TYPE_F32 && buf.rawData != nullptr)
        return static_cast<const float*>(buf.rawData);
    return nullptr;
}

float* asMutableFloatPtr(Buffer& buf) {
    if (buf.dtype == GGML_TYPE_F32 && buf.rawData != nullptr)
        return static_cast<float*>(buf.rawData);
    return nullptr;
}

size_t bufferByteSize(const Buffer& buf) {
    switch (buf.dtype) {
        case GGML_TYPE_F32:  return buf.numElements * sizeof(float);
        case GGML_TYPE_F16:  return buf.numElements * sizeof(uint16_t);
        case GGML_TYPE_I8:   return buf.numElements * sizeof(int8_t);
        case GGML_TYPE_I16:  return buf.numElements * sizeof(int16_t);
        case GGML_TYPE_I32:  return buf.numElements * sizeof(int32_t);
        case GGML_TYPE_I64:  return buf.numElements * sizeof(int64_t);
        case GGML_TYPE_F64:  return buf.numElements * sizeof(double);
        default: break;
    }
    // quantized types: block-structured, treat numElements as block count
    return buf.numElements;  // caller must know block layout
}

std::vector<float> dequantizeToFloat(const Buffer& buf) {
    if (buf.dtype == GGML_TYPE_F32) {
        const float* src = static_cast<const float*>(buf.rawData);
        return std::vector<float>(src, src + buf.numElements);
    }
    if (buf.dtype == GGML_TYPE_F16) {
        const uint16_t* src = static_cast<const uint16_t*>(buf.rawData);
        std::vector<float> out(buf.numElements);
        for (size_t i = 0; i < buf.numElements; ++i)
            out[i] = fp16Tofp32(src[i]);
        return out;
    }
    // quantized types not yet implemented
    throw std::runtime_error("dequantizeToFloat: unsupported dtype");
}

// ─── Tensor loading ───────────────────────────────────────────────────────────

Tensor loadTensor(const char* tensorLocation, bool mut, uint32_t size, uint32_t type) {
    Tensor tensor;
    tensor.mut = mut;

    const char* cursor = tensorLocation;

    if (mut) {
        switch (type) {
            case GGML_TYPE_F32: {
                float* data = static_cast<float*>(std::malloc(size * sizeof(float)));
                if (!data) throw std::bad_alloc();
                std::memcpy(data, cursor, size * sizeof(float));
                tensor.data = Buffer(GGML_TYPE_F32, data, size, true);
                break;
            }
            case GGML_TYPE_F16: {
                float* data = static_cast<float*>(std::malloc(size * sizeof(float)));
                if (!data) throw std::bad_alloc();
                const uint16_t* f16Ptr = reinterpret_cast<const uint16_t*>(cursor);
                for (uint32_t i = 0; i < size; ++i)
                    data[i] = fp16Tofp32(f16Ptr[i]);
                tensor.data = Buffer(GGML_TYPE_F32, data, size, true);
                break;
            }
            // quantized types: store raw view for later dequantize
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q8_1:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_Q8_K:
            default: {
                // store raw bytes; caller must dequantize before math
                tensor.data = Buffer(static_cast<ggml_type>(type),
                               const_cast<char*>(cursor), size, false);
                break;
            }
        }
    } else {
        // read-only view into mmap'd file — no copy
        tensor.data = Buffer(static_cast<ggml_type>(type),
                       const_cast<char*>(cursor),
                       size,
                       false);
    }

    return tensor;
}

// ─── Arithmetic ───────────────────────────────────────────────────────────────

static void requireFloat(const Buffer& buf, const char* name) {
    if (buf.dtype != GGML_TYPE_F32 || buf.rawData == nullptr)
        throw std::runtime_error(
            std::string(name) + " must be a float (F32) buffer");
}

void addTensors(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.dims != b.dims)
        throw std::invalid_argument("Tensors must have the same dimensions for addition.");
    if (out.dims != a.dims)
        throw std::invalid_argument("Output tensor must have the same dimensions as input tensors.");

    requireFloat(a.data, "addTensors: a");
    requireFloat(b.data, "addTensors: b");
    requireFloat(out.data, "addTensors: out");

    const float* pa = static_cast<const float*>(a.data.rawData);
    const float* pb = static_cast<const float*>(b.data.rawData);
    float*       po = static_cast<float*>(out.data.rawData);

    const auto& dims = a.dims;
    size_t ndim = dims.size();
    size_t total = 1;
    for (size_t d = 0; d < ndim; ++d) total *= dims[d];

    std::vector<size_t> idx(ndim, 0);
    for (size_t n = 0; n < total; ++n) {
        size_t offA = 0, offB = 0, offOut = 0;
        for (size_t d = 0; d < ndim; ++d) {
            offA   += idx[d] * a.stride[d];
            offB   += idx[d] * b.stride[d];
            offOut += idx[d] * out.stride[d];
        }
        po[offOut] = pa[offA] + pb[offB];

        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (++idx[d] < dims[d]) break;
            idx[d] = 0;
        }
    }
}

void subTensors(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.dims != b.dims)
        throw std::invalid_argument("Tensors must have the same dimensions for subtraction.");
    if (out.dims != a.dims)
        throw std::invalid_argument("Output tensor must have the same dimensions as input tensors.");

    requireFloat(a.data, "subTensors: a");
    requireFloat(b.data, "subTensors: b");
    requireFloat(out.data, "subTensors: out");

    const float* pa = static_cast<const float*>(a.data.rawData);
    const float* pb = static_cast<const float*>(b.data.rawData);
    float*       po = static_cast<float*>(out.data.rawData);

    const auto& dims = a.dims;
    size_t ndim = dims.size();
    size_t total = 1;
    for (size_t d = 0; d < ndim; ++d) total *= dims[d];

    std::vector<size_t> idx(ndim, 0);
    for (size_t n = 0; n < total; ++n) {
        size_t offA = 0, offB = 0, offOut = 0;
        for (size_t d = 0; d < ndim; ++d) {
            offA   += idx[d] * a.stride[d];
            offB   += idx[d] * b.stride[d];
            offOut += idx[d] * out.stride[d];
        }
        po[offOut] = pa[offA] - pb[offB];

        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (++idx[d] < dims[d]) break;
            idx[d] = 0;
        }
    }
}

void mulTensors(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.dims.size() != 2 || b.dims.size() != 2)
        throw std::invalid_argument("mulTensors currently only supports 2D tensors.");

    uint64_t M = a.dims[0];
    uint64_t K = a.dims[1];
    uint64_t K2 = b.dims[0];
    uint64_t N = b.dims[1];

    if (K != K2)
        throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");
    if (out.dims.size() != 2 || out.dims[0] != M || out.dims[1] != N)
        throw std::invalid_argument("Output tensor must have dimensions (M, N).");

    requireFloat(a.data, "mulTensors: a");
    requireFloat(b.data, "mulTensors: b");
    requireFloat(out.data, "mulTensors: out");

    const float* pa = static_cast<const float*>(a.data.rawData);
    const float* pb = static_cast<const float*>(b.data.rawData);
    float*       po = static_cast<float*>(out.data.rawData);

    uint64_t strideA0 = a.stride[0], strideA1 = a.stride[1];
    uint64_t strideB0 = b.stride[0], strideB1 = b.stride[1];
    uint64_t strideO0 = out.stride[0], strideO1 = out.stride[1];

    // ── thread count ──────────────────────────────────────────────────────

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    if (numThreads > M) numThreads = static_cast<unsigned int>(M);
    if (numThreads < 1) numThreads = 1;

    // ── worker: i → k → j  (k/j swapped for cache-friendly streaming) ────

    auto worker = [&](uint64_t iStart, uint64_t iEnd) {
        for (uint64_t i = iStart; i < iEnd; ++i) {
            float* poRow = po + i * strideO0;

            // zero this output row
            for (uint64_t j = 0; j < N; ++j)
                poRow[j * strideO1] = 0.0f;

            // accumulate: C[i][:] += A[i][k] * B[k][:]
            for (uint64_t k = 0; k < K; ++k) {
                float aik = pa[i * strideA0 + k * strideA1];
                const float* pbRow = pb + k * strideB0;
                for (uint64_t j = 0; j < N; ++j) {
                    poRow[j * strideO1] += aik * pbRow[j * strideB1];
                }
            }
        }
    };

    // ── launch threads ────────────────────────────────────────────────────

    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    uint64_t rowsPerThread = (M + numThreads - 1) / numThreads;

    for (unsigned int t = 0; t < numThreads; ++t) {
        uint64_t start = t * rowsPerThread;
        uint64_t end   = std::min(start + rowsPerThread, M);
        if (start >= end) break;
        threads.emplace_back(worker, start, end);
    }

    for (auto& thr : threads)
        thr.join();
}

// ─── Transpose ────────────────────────────────────────────────────────────────

Tensor transpose(const Tensor& a) {
    Tensor out;
    out.dims = a.dims;
    std::reverse(out.dims.begin(), out.dims.end());
    out.mut = true;

    size_t ndim = out.dims.size();
    out.stride.resize(ndim);
    if (ndim > 0) {
        out.stride[ndim - 1] = 1;
        for (int d = static_cast<int>(ndim) - 2; d >= 0; --d)
            out.stride[d] = out.stride[d + 1] * out.dims[d + 1];
    }

    size_t total = 1;
    for (size_t d = 0; d < ndim; ++d) total *= out.dims[d];

    const float* src = asFloatPtr(a.data);
    bool needsDequant = (src == nullptr);

    std::vector<float> dequantBuf;
    if (needsDequant) {
        dequantBuf = dequantizeToFloat(a.data);
        src = dequantBuf.data();
    }

    float* dst = static_cast<float*>(std::malloc(total * sizeof(float)));
    if (!dst) throw std::bad_alloc();

    std::vector<size_t> idx(ndim, 0);
    for (size_t n = 0; n < total; ++n) {
        size_t offA = 0;
        for (size_t d = 0; d < ndim; ++d)
            offA += idx[d] * a.stride[ndim - 1 - d];

        size_t offOut = 0;
        for (size_t d = 0; d < ndim; ++d)
            offOut += idx[d] * out.stride[d];

        dst[offOut] = src[offA];

        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (++idx[d] < out.dims[d]) break;
            idx[d] = 0;
        }
    }

    out.data = Buffer(GGML_TYPE_F32, dst, total, true);
    return out;
}

// ─── Print ────────────────────────────────────────────────────────────────────

void printTensor(const Tensor& tensor) {
    const auto& dims = tensor.dims;
    size_t ndim = dims.size();

    if (ndim == 0 || tensor.stride.size() != ndim) {
        std::cout << "[]" << std::endl;
        return;
    }

    size_t total = 1;
    for (size_t d = 0; d < ndim; ++d) total *= dims[d];

    if (total == 0) {
        std::cout << "[]" << std::endl;
        return;
    }

    const float* data = asFloatPtr(tensor.data);
    std::vector<float> dequantBuf;

    if (!data) {
        dequantBuf = dequantizeToFloat(tensor.data);
        data = dequantBuf.data();
    }

    std::vector<size_t> idx(ndim, 0);
    for (size_t n = 0; n < total; ++n) {
        size_t off = 0;
        for (size_t d = 0; d < ndim; ++d)
            off += idx[d] * tensor.stride[d];

        std::cout << data[off] << " ";

        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (++idx[d] < dims[d]) break;
            idx[d] = 0;
            if (d > 0) std::cout << "\n";
        }
    }
    std::cout << std::endl;
}

// ─── Sum ──────────────────────────────────────────────────────────────────────

double sum(const Tensor& a) {
    double result = 0.0;

    const float* data = asFloatPtr(a.data);
    std::vector<float> dequantBuf;

    if (!data) {
        dequantBuf = dequantizeToFloat(a.data);
        data = dequantBuf.data();
    }

    for (size_t i = 0; i < a.data.numElements; ++i)
        result += static_cast<double>(data[i]);

    return result;
}

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
