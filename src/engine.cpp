#include "../headers/gguf.h"
#include "../headers/parser.h"
#include <sys/mman.h>
#include <iostream>

int RunEngine() {
    const char* filename = "models/Llama-3.2-1B-Instruct-Q8_0.gguf";
    GGufStarter ggufFile = parseGGUF(filename);    
    return 0;
}