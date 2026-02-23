#include "../headers/gguf.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>

void* loadFile(const char* filename, size_t n_bytes) {
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return nullptr;
    }

    void* mapped = mmap(nullptr, n_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (mapped == MAP_FAILED) {
        std::cerr << "Failed to mmap file" << std::endl;
        return nullptr;
    }

    return mapped;
}
