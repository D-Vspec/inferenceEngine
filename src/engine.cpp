#include "../headers/gguf.h"
#include "../headers/parser.h"
#include <sys/mman.h>
#include <iostream>

int RunEngine() {
    const char* filename = "models/Llama-3.2-1B-Instruct-Q8_0.gguf";
    MappedFile ggufFile = parseGGUF(filename);
    std::cout << "GGUF file loaded successfully: " << filename << std::endl;
    std::cout << "File size: " << ggufFile.size << " bytes" << std::endl;
    
    GGufHeader *header = (GGufHeader*)ggufFile.data;
    std::cout << "Magic: " << std::hex << header->magic << std::dec << std::endl;
    std::cout << "Version: " << header->version << std::endl;
    std::cout << "Tensor count: " << header->tensor_count << std::endl;
    std::cout << "Metadata KV count: " << header->metadata_kv_count << std::endl;
    return 0;
}