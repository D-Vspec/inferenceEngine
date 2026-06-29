#include "../headers/gguf.h"
#include "../headers/parser.h"
#include "../headers/weights.h"

int RunEngine() {
    const char* filename = "models/qwen2-0_5b-instruct-fp16.gguf";
    GGufStarter ggufMetadata = parseGGUF(filename);

    if (ggufMetadata.tensorData == nullptr)
        return 1;

    auto weights = loadWeights(ggufMetadata);

    int numHiddenLayers = static_cast<int>(
        ggufMetadata.metadata_map.at("qwen2.block_count").value.asUint32());

    constexpr int maxNewTokens = 64;

    // TODO: tokenize input prompt, get initial token IDs
    // TODO: embedding lookup → hidden state

    for (int step = 0; step < maxNewTokens; ++step) {
        for (int layer = 0; layer < numHiddenLayers; ++layer) {
            // TODO: rmsNorm → attention → residual
            // TODO: rmsNorm → ffn (swiglu) → residual
        }
        // TODO: final rmsNorm → lm_head → logits
        // TODO: sample next token
        // TODO: embedding lookup for next token
    }

    return 0;
}