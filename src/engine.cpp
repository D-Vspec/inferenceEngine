#include "../headers/gguf.h"
#include "../headers/parser.h"

int RunEngine() {
    const char* filename = "models/qwen2-0_5b-instruct-fp16.gguf";
    GGufStarter ggufMetadata = parseGGUF(filename);

    if (ggufMetadata.tensorData == nullptr)
        return 1;

    // TODO: load all weight tensors
    // TODO: forward pass
    // TODO: sample next token

    return 0;
}