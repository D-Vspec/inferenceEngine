#include "../headers/weights.h"
#include "../headers/tensor.h"

#include <string>
#include <unordered_map>

std::unordered_map<std::string, Tensor> loadWeights(const GGufStarter& gguf) {
    std::unordered_map<std::string, Tensor> weights;
    weights.reserve(gguf.tensor_metadata.size());

    for (const auto& [name, info] : gguf.tensor_metadata) {
        // total logical elements (not bytes)
        size_t numElements = 1;
        for (uint64_t d : info.dims)
            numElements *= d;

        const char* dataPtr = gguf.tensorData + info.offset;

        // mut=false → raw view into mmap'd file, no copy
        Tensor tensor = loadTensor(dataPtr, false,
                                   static_cast<uint32_t>(numElements), info.type);

        // copy dims and compute contiguous row-major strides
        tensor.dims.assign(info.dims.begin(), info.dims.end());
        tensor.stride.resize(tensor.dims.size());
        if (!tensor.dims.empty()) {
            tensor.stride.back() = 1;
            for (int i = static_cast<int>(tensor.dims.size()) - 2; i >= 0; --i)
                tensor.stride[i] = tensor.stride[i + 1] * tensor.dims[i + 1];
        }

        weights.emplace(std::string(name), std::move(tensor));
    }

    return weights;
}
