#include "../headers/parser.h"
#include "../headers/gguf.h"
#include <iostream>
#include <vector>

MappedFile getHeaders(const char* filename) {
    MappedFile ggufFile = loadFile(filename);
    if (ggufFile.data == nullptr) {
        std::cerr << "Failed to load GGUF file: " << filename << std::endl;
        return {};
    }
    return ggufFile;
}

std::pair<std::unordered_map<std::string_view, metadata>, const char*> parseMetadata(const char* cursor, size_t metadata_kv_count){
    std::unordered_map<std::string_view, metadata> metadata_map;

    for(size_t i = 0; i < metadata_kv_count; i++) {
        uint64_t keyLength = *(uint64_t*)cursor;
        cursor += sizeof(uint64_t);

        std::string_view key(cursor, keyLength);
        cursor += keyLength;

        gguf_metadata_value_type valueType = *(gguf_metadata_value_type*)cursor;
        cursor += sizeof(gguf_metadata_value_type);

        metadata value;
        value.value_type = valueType;

        switch (valueType) {
            case GGUF_METADATA_VALUE_TYPE_UINT8:
                value.value = *(uint8_t*)cursor;
                cursor += sizeof(uint8_t);
                break;
            case GGUF_METADATA_VALUE_TYPE_INT8:
                value.value = *(int8_t*)cursor;
                cursor += sizeof(int8_t);
                break;
            case GGUF_METADATA_VALUE_TYPE_UINT16:
                value.value = *(uint16_t*)cursor;
                cursor += sizeof(uint16_t);
                break;
            case GGUF_METADATA_VALUE_TYPE_INT16:
                value.value = *(int16_t*)cursor;
                cursor += sizeof(int16_t);
                break;
            case GGUF_METADATA_VALUE_TYPE_UINT32:
                value.value = *(uint32_t*)cursor;
                cursor += sizeof(uint32_t);
                break;
            case GGUF_METADATA_VALUE_TYPE_INT32:
                value.value = *(int32_t*)cursor;
                cursor += sizeof(int32_t);
                break;
            case GGUF_METADATA_VALUE_TYPE_FLOAT32:
                value.value = *(float*)cursor;
                cursor += sizeof(float);
                break;
            case GGUF_METADATA_VALUE_TYPE_BOOL:
                value.value = *(bool*)cursor;
                cursor += sizeof(bool);
                break;
            case GGUF_METADATA_VALUE_TYPE_UINT64:
                value.value = *(uint64_t*)cursor;
                cursor += sizeof(uint64_t);
                break;
            case GGUF_METADATA_VALUE_TYPE_INT64:
                value.value = *(int64_t*)cursor;
                cursor += sizeof(int64_t);
                break;
            case GGUF_METADATA_VALUE_TYPE_FLOAT64:
                value.value = *(double*)cursor;
                cursor += sizeof(double);
                break; 
            case GGUF_METADATA_VALUE_TYPE_STRING: {
                uint64_t strLength = *(uint64_t*)cursor;
                cursor += sizeof(uint64_t);
                value.value = std::string_view(cursor, strLength);
                cursor += strLength;

                break;
            } 
            case GGUF_METADATA_VALUE_TYPE_ARRAY: {
                gguf_metadata_value_type arrayValueType = *(gguf_metadata_value_type*)cursor;
                cursor += sizeof(gguf_metadata_value_type);
                uint64_t arrayLength = *(uint64_t*)cursor;
                cursor += sizeof(uint64_t); 

                std::cout << "Size of array: " << arrayLength << std::endl;

                switch (arrayValueType) {
                    case GGUF_METADATA_VALUE_TYPE_UINT8: {
                        value.value = std::span<uint8_t>((uint8_t*)cursor, arrayLength);
                        cursor += arrayLength * sizeof(uint8_t);
                        break;
                    }
                    case GGUF_METADATA_VALUE_TYPE_INT8: {
                        value.value = std::span<int8_t>((int8_t*)cursor, arrayLength);
                        cursor += arrayLength * sizeof(int8_t);
                        break;
                    }
                    case GGUF_METADATA_VALUE_TYPE_UINT16: {
                        value.value = std::span<uint16_t>((uint16_t*)cursor, arrayLength);
                        cursor += arrayLength * sizeof(uint16_t);
                        break;
                    }
                    case GGUF_METADATA_VALUE_TYPE_INT16: {
                        value.value = std::span<int16_t>((int16_t*)cursor, arrayLength);
                        cursor += arrayLength * sizeof(int16_t);
                        break;
                    }
                    case GGUF_METADATA_VALUE_TYPE_UINT32: {
                        value.value = std::span<uint32_t>((uint32_t*)cursor, arrayLength);
                        cursor += arrayLength * sizeof(uint32_t);
                        break;
                    }
                    case GGUF_METADATA_VALUE_TYPE_INT32: {
                        value.value = std::span<int32_t>((int32_t*)cursor, arrayLength);
                        cursor += arrayLength * sizeof(int32_t);
                        break;
                    }
                    case GGUF_METADATA_VALUE_TYPE_FLOAT32: {
                        value.value = std::span<float>((float*)cursor, arrayLength);
                        cursor += arrayLength * sizeof(float);
                        break;
                    }
                    case GGUF_METADATA_VALUE_TYPE_BOOL: {
                        value.value = std::span<bool>((bool*)cursor, arrayLength);
                        cursor += arrayLength * sizeof(bool);
                        break;
                    }
                    case GGUF_METADATA_VALUE_TYPE_UINT64: {
                        value.value = std::span<uint64_t>((uint64_t*)cursor, arrayLength);
                        cursor += arrayLength * sizeof(uint64_t);
                        break;
                    }
                    case GGUF_METADATA_VALUE_TYPE_INT64: {
                        value.value = std::span<int64_t>((int64_t*)cursor, arrayLength);
                        cursor += arrayLength * sizeof(int64_t);
                        break;
                    }
                    case GGUF_METADATA_VALUE_TYPE_FLOAT64: {
                        value.value = std::span<double>((double*)cursor, arrayLength);
                        cursor += arrayLength * sizeof(double);
                        break;
                    }
                    case GGUF_METADATA_VALUE_TYPE_STRING: {
                        std::string_view* stringArray = new std::string_view[arrayLength];
                        for (uint64_t j = 0; j < arrayLength; j++) {
                            uint64_t strLength = *(uint64_t*)cursor;
                            cursor += sizeof(uint64_t);
                            stringArray[j] = std::string_view(cursor, strLength);
                            cursor += strLength;

                        }
                        value.value = std::span<std::string_view>(stringArray, arrayLength);
                        break;
                    }
                    default:
                        std::cerr << "Unsupported array value type: " << arrayValueType << std::endl;
                        return {};
                }
                break;
            }
            default:
                std::cerr << "Unsupported metadata value type: " << valueType << std::endl;
                return {};
        } //switch ends here

        metadata_map[key] = value;
    }

    return { std::move(metadata_map), cursor };
}

std::pair<std::unordered_map<std::string_view, TensorInfo>, const char*> getTensorMetadata(const char* cursor, size_t tensorCount){
    std::unordered_map<std::string_view, TensorInfo> tensorMetadata;

    for(size_t i=0;i<tensorCount;i++) {
        uint64_t nameLength = *(uint64_t*)cursor;
        cursor += sizeof(uint64_t);

        std::string_view name = std::string_view(cursor, nameLength);
        cursor += nameLength;

        tensorMetadata[name] = {};

        uint32_t dimCount = *(uint32_t*)cursor;
        cursor += sizeof(uint32_t);

        tensorMetadata[name].dims = std::span<const uint64_t>((const uint64_t*) cursor, dimCount);
        cursor += dimCount * sizeof(uint64_t);

        tensorMetadata[name].type = *(uint32_t*)cursor;
        cursor += sizeof(uint32_t);

        tensorMetadata[name].offset = *(uint64_t*)cursor;
        cursor += sizeof(uint64_t);
    }

    return { std::move(tensorMetadata), cursor };
}

GGufStarter parseGGUF(const char* filename) {

    MappedFile ggufFile = getHeaders(filename);

   
    const char* cursor = (const char*)ggufFile.data;

    if (ggufFile.data == nullptr) {
        std::cerr << "Failed to load GGUF file: " << filename << std::endl;
        return {};
    }

    GGufHeader* header = (GGufHeader*)ggufFile.data;    

    cursor += sizeof(GGufHeader);
    
    std::cout << "Cursor at " << (void *)cursor << " after reading header" << std::endl;

    auto [metadata_map, cursor_after_metadata] = parseMetadata(cursor, header->metadata_kv_count);

    cursor = cursor_after_metadata;

    std::cout << "Cursor at " << (void *)cursor << " after reading kv metadata" << std::endl;

    auto [tensorMetadata, cursor_after_tensors] = getTensorMetadata(cursor, header->tensor_count);

    cursor = cursor_after_tensors;

    std::cout << "Cursor at " << (void *)cursor << " after reading tensor metadata" << std::endl;

    return { *header, std::move(metadata_map), std::move(tensorMetadata) };
}



