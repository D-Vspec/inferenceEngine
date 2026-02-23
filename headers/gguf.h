#ifndef GGUF_H
#define GGUF_H

#include <cstdint>
#include <cstddef>

struct MappedFile {
    void* data;
    size_t size;
};

MappedFile loadFile(const char* filename);

#endif
