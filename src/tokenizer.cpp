#include "../headers/tokenizer.h"
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



