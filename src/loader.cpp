#include "../headers/gguf.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>

MappedFile loadFile(const char* filename) {
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return {nullptr, 0};
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        std::cerr << "Failed to stat file: " << filename << std::endl;
        close(fd);
        return {nullptr, 0};
    }
    size_t file_size = st.st_size;

    std::cout << "File size: " << file_size << " bytes" << std::endl;

    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (mapped == MAP_FAILED) {
        std::cerr << "Failed to mmap file" << std::endl;
        return {nullptr, 0};
    }

    return {mapped, file_size};
}
