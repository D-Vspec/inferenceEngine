#include "../headers/util.h"

#include <cstring>
#include <cstring>

float fp16Tofp32(uint16_t h) {
    uint32_t sign     = (h & 0x8000) << 16;
    uint32_t exponent = (h & 0x7C00) >> 10;
    uint32_t mantissa = (h & 0x03FF) << 13;

    if (exponent == 0x1F) { // Infinity or NaN
        exponent = 0xFF;
    } else if (exponent == 0) { // Zero or Denormal
        if (mantissa != 0) {
            exponent = 127 - 15;
            while (!(mantissa & 0x00800000)) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x007FFFFF;
        }
    } else {
        exponent += (127 - 15);
    }

    uint32_t u32 = sign | (exponent << 23) | mantissa;

    float f;
    std::memcpy(&f, &u32, sizeof(float));
    return f;
}
