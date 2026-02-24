#include "../headers/gguf.h"
#include "../headers/parser.h"
#include <sys/mman.h>
#include <iostream>

int RunEngine() {
    const char* filename = "models/Llama-3.2-1B-Instruct-Q8_0.gguf";
    GGufStarter ggufFile = parseGGUF(filename);    

    for (const auto& [key, meta] : ggufFile.metadata_map) {
        std::cout << "Key: " << key << ", Value Type: " << meta.value_type << ", Value Index: " << meta.value.index() << std::endl;
    }

    std::cout << "Length of metadata map: " << ggufFile.metadata_map.size() << std::endl;

    return 0;
}