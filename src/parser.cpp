#include "../headers/parser.h"
#include "../headers/gguf.h"
#include <iostream>

MappedFile parseGGUF(const char* filename) {
    MappedFile ggufFile = loadFile(filename);
    if (ggufFile.data == nullptr) {
        std::cerr << "Failed to load GGUF file: " << filename << std::endl;
        return {};
    }
    return ggufFile;
}

