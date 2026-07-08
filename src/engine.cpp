#include "../headers/gguf.h"
#include "../headers/parser.h"
#include "../headers/weights.h"
#include "../headers/activations.h"

#include <iostream>
#include <string>
#include <cstring>

int RunEngine() {
    const char* filename = "models/qwen2-0_5b-instruct-fp16.gguf";
    GGufStarter ggufMetadata = parseGGUF(filename);

    if (ggufMetadata.tensorData == nullptr)
        return 1;

    auto weights = loadWeights(ggufMetadata);

    std::string arch = std::string(
        ggufMetadata.metadata_map["general.architecture"].value.asStringView()
    );

    std::string ctxKey = arch + ".context_length";
    uint32_t contextLength =
        ggufMetadata.metadata_map[ctxKey].value.asUint32();

    std::string blockKey = arch + ".block_count";
    uint32_t numLayers =
        ggufMetadata.metadata_map[blockKey].value.asUint32();

    std::string embedKey = arch + ".embedding_length";
    uint32_t embedDim =
        ggufMetadata.metadata_map[embedKey].value.asUint32();

    const Tensor& embedWeights = weights.at("token_embd.weight");
    const float* embedData = asFloatPtr(embedWeights.data);
    float* xData = static_cast<float*>(std::malloc(embedDim * sizeof(float)));
    std::memcpy(xData, embedData, embedDim * sizeof(float));

    Tensor x;
    x.dims = {embedDim};
    x.stride = {1};
    x.mut = true;
    x.data = Buffer(GGML_TYPE_F32, xData, embedDim, true);

    const uint32_t numForwardPasses = contextLength;
    const uint32_t numAttentionRuns = numForwardPasses * numLayers;

    std::cout << "Architecture:     " << arch << std::endl;
    std::cout << "Context length:   " << contextLength << std::endl;
    std::cout << "Layers:           " << numLayers << std::endl;
    std::cout << "Forward passes:   " << numForwardPasses << std::endl;
    std::cout << "Attention runs:   " << numAttentionRuns << std::endl;

    for (uint32_t step = 0; step < numForwardPasses; ++step) {
        for (uint32_t layer = 0; layer < numLayers; ++layer) {
            std::string attnNormKey =
                "blk." + std::to_string(layer) + ".attn_norm.weight";
            const Tensor& attnNormWeight = weights.at(attnNormKey);

            x = rmsNorm(x, attnNormWeight);
        }
    }

    return 0;
}