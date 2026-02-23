#include "headers/gguf.h"
#include <sys/mman.h>
#include <iostream>

int main() {
    size_t n = 1024;
    void* data = loadFile("models/Llama-3.2-1B-Instruct-Q8_0.gguf", n);
    
    GGufHeader* header = static_cast<GGufHeader*>(data); 

    if (data) {
        std::cout << header->magic << std::endl;
        std::cout << header->version << std::endl;
        std::cout << header->tensor_count << std::endl;
        std::cout << header->metadata_kv_count << std::endl;
        munmap(data, n);
    }
    return 0;
}
