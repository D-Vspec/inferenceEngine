#include "../headers/gguf.h"
#include "../headers/parser.h"
#include <sys/mman.h>
#include <iostream>

int RunEngine() {
    const char* filename = "models/Llama-3.2-1B-Instruct-Q8_0.gguf";
    GGufStarter ggufMetadata = parseGGUF(filename);    

    for (const auto& [key, meta] : ggufMetadata.metadata_map) {
        std::cout << "Key: " << key << ", Value Type: " << meta.value_type << ", Value Index: " << meta.value.index() << std::endl;
    }

    std::cout << "Length of kv metadata map: " << ggufMetadata.metadata_map.size() << std::endl;

    std::cout << "Tensor count: " << ggufMetadata.tensor_metadata.size() << std::endl;

     for (const auto& tensorInfo : ggufMetadata.tensor_metadata) {
        std::cout << "Tensor Name: " << tensorInfo.name << ", Type: " << tensorInfo.type << ", Offset: " << tensorInfo.offset << std::endl;
        std::cout << "Dimensions: ";
        for (const auto& dim : tensorInfo.dims) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}