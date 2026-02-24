#include "../headers/parser.h"
#include "../headers/gguf.h"
#include <iostream>

MappedFile getHeaders(const char* filename) {
    MappedFile ggufFile = loadFile(filename);
    if (ggufFile.data == nullptr) {
        std::cerr << "Failed to load GGUF file: " << filename << std::endl;
        return {};
    }
    return std::move(ggufFile);
}

std::unordered_map<std::string_view, metadata> parseMetadata(const char* cursor, size_t metadata_kv_count){
    std::unordered_map<std::string_view, metadata> metadata_map;

    for(size_t i = 0; i < metadata_kv_count; i++) {
        uint64_t keyLength = *(uint64_t*)cursor;
        cursor += sizeof(uint64_t);

        std::string_view key(cursor, keyLength);
        cursor += keyLength;

        std::cout << "Parsing metadata key: " << key << std::endl;

        gguf_metadata_value_type valueType = *(gguf_metadata_value_type*)cursor;
        cursor += sizeof(gguf_metadata_value_type);

        std::cout << "Value type: " << valueType << std::endl;
        
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

                std::cout << "Parsed string value: " << std::get<std::string_view>(value.value) << std::endl;

                break;
            } 
            case GGUF_METADATA_VALUE_TYPE_ARRAY: {
                std::cout << "Parsing array value..." << std::endl;
                gguf_metadata_value_type arrayValueType = *(gguf_metadata_value_type*)cursor;
                cursor += sizeof(gguf_metadata_value_type);
                std::cout << "Array value type: " << arrayValueType << std::endl;
                uint64_t arrayLength = *(uint64_t*)cursor;
                cursor += sizeof(uint64_t); 
                std::cout << "Array length: " << arrayLength << std::endl;
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

                            std::cout << "Parsed string array element: " << stringArray[j] << std::endl;
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

        std::cout << "Parsed metadata key: " << key << " with value type: " << valueType << std::endl;
        std::cout << "Value : " << value.value.index() << std::endl;

        metadata_map[key] = value;
    }

    return metadata_map;
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

    std::unordered_map<std::string_view, metadata> metadata_map = parseMetadata(cursor, header->metadata_kv_count);

    return { *header, std::move(metadata_map) };
}

