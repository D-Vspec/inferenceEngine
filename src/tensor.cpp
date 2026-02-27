#include "../headers/tensor.h"
#include "../headers/parser.h"
#include "../headers/util.h"

Tensor loadTensor(const char* tensorLocation, bool mut, uint32_t size, uint32_t type) {
    Tensor tensor;
    tensor.mut = mut;

    const char* cursor = tensorLocation;

    switch (type) {
        case GGML_TYPE_F32: {
            if(tensor.mut == true){
                std::vector<float> data(size);
                for(uint32_t i = 0; i < size; i++){
                    data[i] = *(float*)cursor;
                    cursor += sizeof(float);
                }
                tensor.data = std::move(data);
            }
        }
        case GGML_TYPE_F16: {
            if(tensor.mut == true){
                std::vector<float> data(size);
                const uint16_t* f16_ptr = reinterpret_cast<const uint16_t*>(cursor);
                for(uint32_t i = 0; i < size; i++){
                    data[i] = fp16Tofp32(f16_ptr[i]);
                }
                tensor.data = std::move(data);
            }
            break;
        }
    }

    return tensor;
}