#include "headers/gguf.h"
#include <sys/mman.h>
#include <iostream>

int main() {
    size_t file_size = 0;
    void* data = loadFile("models/Llama-3.2-1B-Instruct-Q8_0.gguf", file_size);

    GGufHeader* header = static_cast<GGufHeader*>(data);

    if (data) {
        std::cout << header->magic << std::endl;
        std::cout << header->version << std::endl;
        std::cout << header->tensor_count << std::endl;
        std::cout << header->metadata_kv_count << std::endl;
        munmap(data, file_size);
    }
    return 0;
}
