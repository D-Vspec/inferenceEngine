#include "../headers/tokenizer.h"
#include "../headers/tensor.h"

#include <iostream>
#include <vector>
#include <queue>

std::unordered_map<std::string_view, uint32_t> buildVocab(std::span<std::string_view> tokenList){

    std::unordered_map<std::string_view, uint32_t> vocab;
    uint32_t index = 0;

    for (auto& token : tokenList) {
        vocab[token] = index++; 
    }
    return vocab;
}

std::vector<std::string> preTokenize(const std::string& input) {
    std::vector<std::string> tokens;
    const size_t n = input.size();
    size_t i = 0;

    while (i < n) {
        size_t start = i;

        if (input[i] == ' ') {
            ++i;
            while (i < n && input[i] == ' ') ++i;
        }

        while (i < n && input[i] != ' ') ++i;

        if (i > start) {
            tokens.emplace_back(input.substr(start, i - start));
        }
    }
    return tokens;
}

std::vector<uint64_t> tokenize(const std::string& input, const std::unordered_map<std::string_view, uint32_t>& vocab){
    std::cout << "Tokenizing: " << input << std::endl;
    std::vector<std::string> preTokens = preTokenize(input);
    std::vector<uint64_t> tokenIds;

    for (const auto& token : preTokens) {
        std::string tempToken;
        for (char c : token) {
            if (c == ' ')
                tempToken += "\xC4\xA0"; // Ġ (U+0120)
            else
                tempToken += c;
        }

        while (!tempToken.empty()) {
            bool found = false;
            for (size_t len = tempToken.size(); len >= 1; --len) {
                std::string prefix = tempToken.substr(0, len);
                if (vocab.find(prefix) != vocab.end()) {
                    std::cout << "  Matched: \"" << prefix << "\" -> " << vocab.at(prefix) << std::endl;
                    tokenIds.push_back(vocab.at(prefix));
                    tempToken = tempToken.substr(len);
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cerr << "Unknown token: " << tempToken[0] << std::endl;
                tempToken = tempToken.substr(1);
            }
        }
    }

    return tokenIds;
}

Tensor tokenToTensor(const uint64_t token, const char* embedWeights, uint32_t type, uint32_t embedSize){
    Tensor tensor;
    tensor.mut = true;

    size_t unitSize;

    switch (type) {
        case GGML_TYPE_F32:
            unitSize = sizeof(float);
            break;
        case GGML_TYPE_F16:
            unitSize = sizeof(uint16_t);
            break;
        default:
            std::cerr << "Unsupported type: " << type << std::endl;
            return {};
    }

    const char* tensorLocation = embedWeights + (token * unitSize);

    tensor = loadTensor(tensorLocation, true, embedSize, type);

    return tensor;
}

std::vector<Tensor> tokensToTensors(const std::vector<uint64_t>& tokens, const char* embedWeights, uint32_t type, uint32_t embedSize){
    std::vector<Tensor> tensors;
    for(const auto& token : tokens){
        tensors.push_back(tokenToTensor(token, embedWeights, type, embedSize));
    }
    return tensors;
}



