#include "../headers/gguf.h"
#include "../headers/parser.h"
#include "../headers/tokenizer.h"
#include <sys/mman.h>
#include <iostream>

int RunEngine() {
    const char* filename = "models/qwen2-0_5b-instruct-fp16.gguf";
    GGufStarter ggufMetadata = parseGGUF(filename);    

    for (const auto& [key, meta] : ggufMetadata.metadata_map) {
        std::cout << "Key: " << key << ", Value Type: " << meta.value_type << ", Value Index: " << meta.value.index() << std::endl;
    }

    std::cout << "Length of kv metadata map: " << ggufMetadata.metadata_map.size() << std::endl;

    std::cout << "Tensor count: " << ggufMetadata.tensor_metadata.size() << std::endl;
    
    //  for (const auto& [name, tensorInfo] : ggufMetadata.tensor_metadata) {
    //     std::cout << "Tensor Name: " << name << ", Type: " << tensorInfo.type << ", Offset: " << tensorInfo.offset << std::endl;
        // std::cout << "Dimensions: ";
        // for (const auto& dim : tensorInfo.dims) {
        //     std::cout << dim << " ";
        // }
        // std::cout << std::endl;
    // }

    std::unordered_map<std::string_view, uint32_t>tokenLookup = buildVocab(std::get<std::span<std::string_view>>(ggufMetadata.metadata_map["tokenizer.ggml.tokens"].value));

    std::vector<uint64_t> tokens = tokenize("My name is Indigo Montoya, you killed my father, prepare to die.", tokenLookup);

    uint32_t embeddingLength = std::get<std::uint32_t>(ggufMetadata.metadata_map["qwen2.embedding_length"].value);
    
    std::cout << "Type of tensor : " << ggufMetadata.tensor_metadata["token_embd.weight"].type << std::endl;

    const char* embedWeights = ggufMetadata.tensorData + ggufMetadata.tensor_metadata["token_embd.weight"].offset;
    std::vector<Tensor> tokenTensors = tokensToTensors(tokens, embedWeights, ggufMetadata.tensor_metadata["token_embd.weight"].type, embeddingLength);

    for(size_t i = 0; i < tokenTensors.size(); i++){
        std::cout << "Token: " << tokens[i] << ", First 5 values of embedding: ";
        auto& data = std::get<std::vector<float>>(tokenTensors[i].data);
        for(size_t j = 0; j < 5 && j < data.size(); j++){
            std::cout << data[j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}