#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <unordered_map>
#include <string_view>
#include <string>
#include <vector>
#include <cstdint>
#include <span>

#include "tensor.h"

std::unordered_map<std::string_view, uint32_t> buildVocab(std::span<std::string_view> tokenList);
std::vector<std::string> preTokenize(const std::string& input);
std::vector<uint64_t> tokenize(const std::string& input, const std::unordered_map<std::string_view, uint32_t>& vocab);
Tensor tokenToTensor(const uint64_t token, const char* embedWeights, uint32_t type, uint32_t embedSize);
std::vector<Tensor> tokensToTensors(const std::vector<uint64_t>& tokens, const char* embedWeights, uint32_t type, uint32_t embedSize);

#endif