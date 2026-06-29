#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "tensor.h"

// ─── Normalization ────────────────────────────────────────────────────────────

Tensor rmsNorm(const Tensor& x, const Tensor& weight, float epsilon = 1e-6f);

// ─── Activations ──────────────────────────────────────────────────────────────

Tensor relu(const Tensor& x);
Tensor silu(const Tensor& x);
Tensor swiglu(const Tensor& gate, const Tensor& up);

#endif
