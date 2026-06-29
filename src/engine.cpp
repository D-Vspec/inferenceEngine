#include "../headers/gguf.h"
#include "../headers/parser.h"
#include "../headers/weights.h"

int RunEngine() {
    const char* filename = "models/qwen2-0_5b-instruct-fp16.gguf";
    GGufStarter ggufMetadata = parseGGUF(filename);

    if (ggufMetadata.tensorData == nullptr)
        return 1;

    auto weights = loadWeights(ggufMetadata);

    constexpr int maxNewTokens = 64;

    // TODO: tokenize input prompt, get initial token IDs
    // TODO: embedding lookup → hidden state

    for (int step = 0; step < maxNewTokens; ++step) {
        // TODO: for each transformer layer:
        //   rmsNorm → attention → residual
        //   rmsNorm → ffn (swiglu) → residual
        // TODO: final rmsNorm → lm_head → logits
        // TODO: sample next token
        // TODO: embedding lookup for next token
    }

    return 0;
}