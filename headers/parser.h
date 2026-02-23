#ifndef PARSER_H
#define PARSER_H

#include <cstdint>
#include <cstddef>
#include <string_view>
#include <span>
#include <variant>
#include <unordered_map>

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
} GGufHeader;

enum ggml_type: uint32_t { //Copied straight from ggml.h :P
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    GGML_TYPE_MXFP4   = 39, // MXFP4 (1 block)
    GGML_TYPE_COUNT   = 40,
};

enum gguf_metadata_value_type: uint32_t { // Also copied from ggml.h because i cant be bothered to make my own enum for this
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
    GGUF_METADATA_VALUE_TYPE_INT8 = 1,
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
    GGUF_METADATA_VALUE_TYPE_INT16 = 3,
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
    GGUF_METADATA_VALUE_TYPE_INT32 = 5,
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
    GGUF_METADATA_VALUE_TYPE_BOOL = 7,
    GGUF_METADATA_VALUE_TYPE_STRING = 8,
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
    GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
    GGUF_METADATA_VALUE_TYPE_INT64 = 11,
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
};

using gguf_metadata_value_t = std::variant<

    uint8_t,
    int8_t,
    uint16_t,
    int16_t,
    uint32_t,
    int32_t,
    float,
    bool,
    std::string_view,
    uint64_t,
    int64_t,
    double,

    std::span<uint8_t>,
    std::span<int8_t>,
    std::span<uint16_t>,
    std::span<int16_t>,
    std::span<uint32_t>,
    std::span<int32_t>,
    std::span<float>,
    std::span<bool>,
    std::span<uint64_t>,
    std::span<int64_t>,
    std::span<double>,

    std::span<std::string_view>,
>;


typedef struct {
    gguf_metadata_value_type value_type;
    gguf_metadata_value_t value;
} metadata;

std::unordered_map<std::string_view, metadata> parseMetadata(const char* data, size_t metadata_kv_count);

#endif
