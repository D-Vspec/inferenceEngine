#ifndef WEIGHTS_H
#define WEIGHTS_H

#include "tensor.h"
#include "parser.h"

#include <string>
#include <unordered_map>

// Load all weight tensors from a parsed GGUF file.
// Returns a map from tensor name → Tensor (read-only views into the mmap'd file).
std::unordered_map<std::string, Tensor> loadWeights(const GGufStarter& gguf);

#endif
