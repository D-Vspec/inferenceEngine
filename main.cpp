#include "headers/gguf.h"
#include "headers/parser.h"
#include <sys/mman.h>
#include <iostream>

int main() {
    const char* filename = "models/Llama-3.2-1B-Instruct-Q8_0.gguf";
    MappedFile ggufFile = parseGGUF(filename);
    std::cout << "GGUF file loaded successfully: " << filename << std::endl;
    std::cout << "File size: " << ggufFile.size << " bytes" << std::endl;

    return 0;
}
