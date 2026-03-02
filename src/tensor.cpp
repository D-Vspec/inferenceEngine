#include "../headers/tensor.h"
#include "../headers/parser.h"
#include "../headers/util.h"

#include <iostream>
#include <stdexcept>

Tensor loadTensor(const char* tensorLocation, bool mut, uint32_t size, uint32_t type) {
    Tensor tensor;
    tensor.mut = mut;

    const char* cursor = tensorLocation;

    switch (type) {
        case GGML_TYPE_F32: {
            if(tensor.mut == true){
                std::vector<float> data(size);
                for(uint32_t i = 0; i < size; i++){
                    data[i] = *(float*)cursor;
                    cursor += sizeof(float);
                }
                tensor.data = std::move(data);
            }
            break;
        }
        case GGML_TYPE_F16: {
            if(tensor.mut == true){
                std::vector<float> data(size);
                const uint16_t* f16_ptr = reinterpret_cast<const uint16_t*>(cursor);
                for(uint32_t i = 0; i < size; i++){
                    data[i] = fp16Tofp32(f16_ptr[i]);
                }
                tensor.data = std::move(data);
            }
            break;
        }
    }

    return tensor;
}

void addTensors(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.dims != b.dims) {
        throw std::invalid_argument("Tensors must have the same dimensions for addition.");
    }

    if (out.dims != a.dims) {
        throw std::invalid_argument("Output tensor must have the same dimensions as input tensors.");
    }

    const auto& dims = a.dims;
    size_t ndim = dims.size();
    size_t total = 1;
    for (size_t d = 0; d < ndim; ++d) total *= dims[d];

    std::visit([&](auto&& data_a, auto&& data_b) {
        std::visit([&](auto&& data_out) {
            using ElemRef = decltype(data_out[0]);
            if constexpr (!std::is_const_v<std::remove_reference_t<ElemRef>>) {
                using OutType = typename std::decay_t<decltype(data_out)>::value_type;

                std::vector<size_t> idx(ndim, 0);

                for (size_t n = 0; n < total; ++n) {
                    size_t off_a = 0, off_b = 0, off_out = 0;
                    for (size_t d = 0; d < ndim; ++d) {
                        off_a   += idx[d] * a.stride[d];
                        off_b   += idx[d] * b.stride[d];
                        off_out += idx[d] * out.stride[d];
                    }

                    data_out[off_out] = static_cast<OutType>(data_a[off_a] + data_b[off_b]);

                    for (int d = ndim - 1; d >= 0; --d) {
                        if (++idx[d] < dims[d]) break;
                        idx[d] = 0;
                    }
                }
            }
        }, out.data);
    }, a.data, b.data);
}

void subTensors(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.dims != b.dims) {
        throw std::invalid_argument("Tensors must have the same dimensions for subtraction.");
    }

    if (out.dims != a.dims) {
        throw std::invalid_argument("Output tensor must have the same dimensions as input tensors.");
    }

    const auto& dims = a.dims;
    size_t ndim = dims.size();
    size_t total = 1;
    for (size_t d = 0; d < ndim; ++d) total *= dims[d];

    std::visit([&](auto&& data_a, auto&& data_b) {
        std::visit([&](auto&& data_out) {
            using ElemRef = decltype(data_out[0]);
            if constexpr (!std::is_const_v<std::remove_reference_t<ElemRef>>) {
                using OutType = typename std::decay_t<decltype(data_out)>::value_type;

                std::vector<size_t> idx(ndim, 0);

                for (size_t n = 0; n < total; ++n) {
                    size_t off_a = 0, off_b = 0, off_out = 0;
                    for (size_t d = 0; d < ndim; ++d) {
                        off_a   += idx[d] * a.stride[d];
                        off_b   += idx[d] * b.stride[d];
                        off_out += idx[d] * out.stride[d];
                    }

                    data_out[off_out] = static_cast<OutType>(data_a[off_a] - data_b[off_b]);

                    for (int d = ndim - 1; d >= 0; --d) {
                        if (++idx[d] < dims[d]) break;
                        idx[d] = 0;
                    }
                }
            }
        }, out.data);
    }, a.data, b.data);
}

void mulTensors(const Tensor& a, const Tensor& b, Tensor& out) {
    // Matrix multiplication: A(M, K) × B(K, N) = C(M, N)
    if (a.dims.size() != 2 || b.dims.size() != 2) {
        throw std::invalid_argument("mulTensors currently only supports 2D tensors.");
    }

    uint64_t M = a.dims[0];
    uint64_t K = a.dims[1];
    uint64_t K2 = b.dims[0];
    uint64_t N = b.dims[1];

    if (K != K2) {
        throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");
    }

    if (out.dims.size() != 2 || out.dims[0] != M || out.dims[1] != N) {
        throw std::invalid_argument("Output tensor must have dimensions (M, N).");
    }

    std::visit([&](auto&& data_a, auto&& data_b) {
        std::visit([&](auto&& data_out) {
            using ElemRef = decltype(data_out[0]);
            if constexpr (!std::is_const_v<std::remove_reference_t<ElemRef>>) {
                using OutType = typename std::decay_t<decltype(data_out)>::value_type;

                for (uint64_t i = 0; i < M; ++i) {
                    for (uint64_t j = 0; j < N; ++j) {
                        OutType sum = 0;
                        for (uint64_t k = 0; k < K; ++k) {
                            size_t off_a = i * a.stride[0] + k * a.stride[1];
                            size_t off_b = k * b.stride[0] + j * b.stride[1];
                            sum += static_cast<OutType>(data_a[off_a] * data_b[off_b]);
                        }
                        size_t off_out = i * out.stride[0] + j * out.stride[1];
                        data_out[off_out] = sum;
                    }
                }
            }
        }, out.data);
    }, a.data, b.data);
}

Tensor transpose(const Tensor &a){
    Tensor out;
    out.dims = a.dims;
    std::reverse(out.dims.begin(), out.dims.end());
    out.mut = true;

    // Compute contiguous strides for the transposed shape
    size_t ndim = out.dims.size();
    out.stride.resize(ndim);
    if (ndim > 0) {
        out.stride[ndim - 1] = 1;
        for (int d = ndim - 2; d >= 0; --d) {
            out.stride[d] = out.stride[d + 1] * out.dims[d + 1];
        }
    }

    size_t total = 1;
    for (size_t d = 0; d < ndim; ++d) total *= out.dims[d];

    std::visit([&](auto&& data_a) {
        using DataType = typename std::decay_t<decltype(data_a)>::value_type;
        std::vector<DataType> data_out(total);

        std::vector<size_t> idx(ndim, 0);
        for (size_t n = 0; n < total; ++n) {
            size_t off_a = 0;
            for (size_t d = 0; d < ndim; ++d) {
                off_a += idx[d] * a.stride[ndim - 1 - d];
            }

            size_t off_out = 0;
            for (size_t d = 0; d < ndim; ++d) {
                off_out += idx[d] * out.stride[d];
            }

            data_out[off_out] = static_cast<DataType>(data_a[off_a]);

            for (int d = ndim - 1; d >= 0; --d) {
                if (++idx[d] < out.dims[d]) break;
                idx[d] = 0;
            }
        }

        out.data = std::move(data_out);
    }, a.data);

    return out;
}

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

    std::visit([&](auto&& data) {
        if (data.size() == 0) {
            std::cout << "[]" << std::endl;
            return;
        }

        std::vector<size_t> idx(ndim, 0);

        for (size_t n = 0; n < total; ++n) {
            size_t off = 0;
            for (size_t d = 0; d < ndim; ++d) {
                off += idx[d] * tensor.stride[d];
            }

            std::cout << data[off] << " ";

            for (int d = ndim - 1; d >= 0; --d) {
                if (++idx[d] < dims[d]) break;
                idx[d] = 0;
                if (d > 0) std::cout << "\n";
            }
        }
        std::cout << std::endl;
    }, tensor.data);
}

double sum(const Tensor& a) {
    double result = 0.0;
    std::visit([&](auto&& data_a) {
        for (const auto& val : data_a) {
            result += static_cast<double>(val);
        }
    }, a.data);
    return result;
}