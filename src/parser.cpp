#include "../headers/parser.h"
#include "../headers/gguf.h"
#include <iostream>
#include <vector>
#include <tuple>

MappedFile getHeaders(const char* filename) {
    MappedFile ggufFile = loadFile(filename);
    if (ggufFile.data == nullptr) {
        std::cerr << "Failed to load GGUF file: " << filename << std::endl;
        return {};
    }
    return ggufFile;
}

std::tuple<std::unordered_map<std::string_view, metadata>, const char*, size_t> parseMetadata(const char* cursor, size_t metadata_kv_count){
    std::unordered_map<std::string_view, metadata> metadata_map;

    for(size_t i = 0; i < metadata_kv_count; i++) {
        uint64_t keyLength = *(uint64_t*)cursor;
        cursor += sizeof(uint64_t);

        std::string_view key(cursor, keyLength);
        cursor += keyLength;

        gguf_metadata_value_type valueType = *(gguf_metadata_value_type*)cursor;
        cursor += sizeof(gguf_metadata_value_type);

        metadata metaVal;
        metaVal.value_type = valueType;
        metaVal.value.valueType = valueType;

        switch (valueType) {
            case GGUF_METADATA_VALUE_TYPE_UINT8:
                metaVal.value.scalar.uint8Val = *(uint8_t*)cursor;
                cursor += sizeof(uint8_t);
                break;
            case GGUF_METADATA_VALUE_TYPE_INT8:
                metaVal.value.scalar.int8Val = *(int8_t*)cursor;
                cursor += sizeof(int8_t);
                break;
            case GGUF_METADATA_VALUE_TYPE_UINT16:
                metaVal.value.scalar.uint16Val = *(uint16_t*)cursor;
                cursor += sizeof(uint16_t);
                break;
            case GGUF_METADATA_VALUE_TYPE_INT16:
                metaVal.value.scalar.int16Val = *(int16_t*)cursor;
                cursor += sizeof(int16_t);
                break;
            case GGUF_METADATA_VALUE_TYPE_UINT32:
                metaVal.value.scalar.uint32Val = *(uint32_t*)cursor;
                cursor += sizeof(uint32_t);
                break;
            case GGUF_METADATA_VALUE_TYPE_INT32:
                metaVal.value.scalar.int32Val = *(int32_t*)cursor;
                cursor += sizeof(int32_t);
                break;
            case GGUF_METADATA_VALUE_TYPE_FLOAT32:
                metaVal.value.scalar.float32Val = *(float*)cursor;
                cursor += sizeof(float);
                break;
            case GGUF_METADATA_VALUE_TYPE_BOOL:
                metaVal.value.scalar.boolVal = *(bool*)cursor;
                cursor += sizeof(bool);
                break;
            case GGUF_METADATA_VALUE_TYPE_UINT64:
                metaVal.value.scalar.uint64Val = *(uint64_t*)cursor;
                cursor += sizeof(uint64_t);
                break;
            case GGUF_METADATA_VALUE_TYPE_INT64:
                metaVal.value.scalar.int64Val = *(int64_t*)cursor;
                cursor += sizeof(int64_t);
                break;
            case GGUF_METADATA_VALUE_TYPE_FLOAT64:
                metaVal.value.scalar.float64Val = *(double*)cursor;
                cursor += sizeof(double);
                break; 
            case GGUF_METADATA_VALUE_TYPE_STRING: {
                uint64_t strLength = *(uint64_t*)cursor;
                cursor += sizeof(uint64_t);
                metaVal.value.stringVal = std::string_view(cursor, strLength);
                cursor += strLength;
                break;
            } 
            case GGUF_METADATA_VALUE_TYPE_ARRAY: {
                gguf_metadata_value_type arrayValueType = *(gguf_metadata_value_type*)cursor;
                cursor += sizeof(gguf_metadata_value_type);
                uint64_t arrayLength = *(uint64_t*)cursor;
                cursor += sizeof(uint64_t); 

                switch (arrayValueType) {
                    case GGUF_METADATA_VALUE_TYPE_STRING: {
                        std::string_view* stringArray = new std::string_view[arrayLength];
                        for (uint64_t j = 0; j < arrayLength; j++) {
                            uint64_t strLength = *(uint64_t*)cursor;
                            cursor += sizeof(uint64_t);
                            stringArray[j] = std::string_view(cursor, strLength);
                            cursor += strLength;
                        }
                        metaVal.value.arrayData = stringArray;
                        metaVal.value.arrayLength = arrayLength;
                        break;
                    }
                    default: {
                        // All non-string array types: point directly into mmap'd file
                        metaVal.value.arrayData = cursor;
                        metaVal.value.arrayLength = arrayLength;
                        // advance cursor by element count * element size
                        size_t elemSize = 0;
                        switch (arrayValueType) {
                            case GGUF_METADATA_VALUE_TYPE_UINT8:  elemSize = 1; break;
                            case GGUF_METADATA_VALUE_TYPE_INT8:   elemSize = 1; break;
                            case GGUF_METADATA_VALUE_TYPE_UINT16: elemSize = 2; break;
                            case GGUF_METADATA_VALUE_TYPE_INT16:  elemSize = 2; break;
                            case GGUF_METADATA_VALUE_TYPE_UINT32: elemSize = 4; break;
                            case GGUF_METADATA_VALUE_TYPE_INT32:  elemSize = 4; break;
                            case GGUF_METADATA_VALUE_TYPE_FLOAT32:elemSize = 4; break;
                            case GGUF_METADATA_VALUE_TYPE_BOOL:   elemSize = 1; break;
                            case GGUF_METADATA_VALUE_TYPE_UINT64: elemSize = 8; break;
                            case GGUF_METADATA_VALUE_TYPE_INT64:  elemSize = 8; break;
                            case GGUF_METADATA_VALUE_TYPE_FLOAT64:elemSize = 8; break;
                            default: break;
                        }
                        cursor += arrayLength * elemSize;
                        break;
                    }
                }
                break;
            }
            default:
                std::cerr << "Unsupported metadata value type: " << valueType << std::endl;
                return {};
        }

        metadata_map[key] = metaVal;
    }

    size_t alignment = 0;
    //Alignment
    if (metadata_map.count("general.alignment") != 0)
        alignment = metadata_map["general.alignment"].value.asUint32();
    else
        alignment = 32;
    
    std::cout << "Alignment: " << alignment << std::endl;

    return { std::move(metadata_map), cursor, alignment };
}

std::pair<std::unordered_map<std::string_view, TensorInfo>, const char*> getTensorMetadata(const char* cursor, size_t tensorCount, size_t alignment){
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

const char* getAlignment(const char* cursor, size_t alignment) {
    uintptr_t ptr = reinterpret_cast<uintptr_t>(cursor);
    size_t offset = (alignment - (ptr % alignment)) % alignment;
    return cursor + offset;
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

    auto [metadata_map, cursor_after_metadata, alignment] = parseMetadata(cursor, header->metadata_kv_count);

    cursor = cursor_after_metadata;

    std::cout << "Cursor at " << (void *)cursor << " after reading kv metadata" << std::endl;

    auto [tensorMetadata, cursor_after_tensors] = getTensorMetadata(cursor, header->tensor_count, alignment);

    cursor = cursor_after_tensors;

    std::cout << "Cursor at " << (void *)cursor << " after reading tensor metadata" << std::endl;

    cursor = getAlignment(cursor, alignment);

    std::cout << "Cursor at " << (void *)cursor << " after alignment" << std::endl;

    return { *header, std::move(metadata_map), std::move(tensorMetadata), cursor };
}




