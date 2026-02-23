#ifndef GGUF_H
#define GGUF_H

#include <cstdint>
#include <cstddef>

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
} GGufHeader;

void* loadFile(const char* filename, size_t n_bytes);

#endif
