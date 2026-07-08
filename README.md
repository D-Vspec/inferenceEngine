# InferenceEngine

A from-scratch C++ inference engine for running transformer language models in the [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) format. No external ML libraries — just the C++ standard library and raw tensor math.

## Supported Models

- Qwen2 (tested with `qwen2-0_5b-instruct-fp16.gguf`)
- Llama 3 (planned — `Llama-3.2-1B-Instruct-Q8_0.gguf` included in `models/`)

## Architecture

```
main.cpp          Entry point
├── parser.cpp    GGUF header + metadata + tensor-info parsing
├── loader.cpp    Memory-mapped file I/O
├── weights.cpp   Tensor extraction from mmap'd region
├── tensor.cpp    Tensor buffers, dequantization, arithmetic ops
├── activations.cpp  RMS norm, ReLU, SiLU, SwiGLU
├── tokenizer.cpp    BPE-style tokenizer with vocab lookup
├── engine.cpp       Main inference loop
└── util.cpp         Miscellaneous helpers
```

## Build

```bash
make
```

Requires **g++** with **C++23** support. The binary is output as `./inference`.

```bash
make clean   # remove build artifacts
```

## Quick Start

1. Place a GGUF model in `models/`
2. Update the filename in `src/engine.cpp` if needed
3. Build and run:

```bash
make
./inference
```

## Project Status

This is an active work-in-progress. See [`TODO`](TODO) for the full task list.

### Done
- GGUF parsing (header, metadata, tensor info)
- Weight loading (zero-copy views into mmap)
- Basic BPE tokenizer
- RMS normalization, ReLU, SiLU, SwiGLU
- Tensor ops: add, multiply, subtract, transpose, sum, dequantize
- Engine loop scaffolding with context-length and layer-count discovery

### In Progress
- Full transformer block (Q/K/V projections, attention, FFN)
- Rotary position embeddings (RoPE)
- KV cache
- Token sampling (greedy, temperature, top-k, top-p)

## Design Decisions

- **Pure STL** — no BLAS, no OpenMP, no external deps beyond libc
- **Zero-copy weights** — tensors are read-only views over the mmap'd GGUF file; only activations allocate new memory
- **C++23** — uses `std::span`, `std::string_view`, and structured bindings throughout
- **Multithreading** — planned via `std::thread` with row-wise matrix partitioning (see [`docs/multithreading.md`](docs/multithreading.md))

## License

MIT
