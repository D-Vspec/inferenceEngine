#include "../headers/gguf.h"
#include "../headers/parser.h"
#include "../headers/weights.h"
#include "../headers/tokenizer.h"

#include <iostream>
#include <string>

int RunEngine() {
    const char* filename = "models/qwen2-0_5b-instruct-fp16.gguf";
    GGufStarter ggufMetadata = parseGGUF(filename);

    if (ggufMetadata.tensorData == nullptr)
        return 1;

    auto weights = loadWeights(ggufMetadata);

    int numHiddenLayers = static_cast<int>(
        ggufMetadata.metadata_map.at("qwen2.block_count").value.asUint32());
    uint32_t embedDim = ggufMetadata.metadata_map.at("qwen2.embedding_length").value.asUint32();

    constexpr int maxNewTokens = 64;

    // ── tokenize input prompt ─────────────────────────────────────────────

    auto vocab = buildVocab(
        ggufMetadata.metadata_map.at("tokenizer.ggml.tokens").value.asStringViewSpan());

    std::string prompt = "Hello, world!";
    std::vector<uint64_t> tokenIds = tokenize(prompt, vocab);

    // ── embedding lookup → initial hidden state ───────────────────────────

    Tensor& embedWeight = weights.at("token_embd.weight");
    const char* embedPtr  = static_cast<const char*>(embedWeight.data.rawData);
    uint32_t    embedType = ggufMetadata.tensor_metadata.at("token_embd.weight").type;

    std::vector<Tensor> embeddings = tokensToTensors(tokenIds, embedPtr, embedType, embedDim);
    Tensor hiddenState = std::move(embeddings.back());  // last token for now

    // ── generation loop ───────────────────────────────────────────────────

    for (int step = 0; step < maxNewTokens; ++step) {
        for (int layer = 0; layer < numHiddenLayers; ++layer) {
            // TODO: rmsNorm → attention → residual
            // TODO: rmsNorm → ffn (swiglu) → residual
        }
        // TODO: final rmsNorm → lm_head → logits
        // TODO: sample next token → update hiddenState
    }

    return 0;
}