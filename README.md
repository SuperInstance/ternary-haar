# ternary-haar

Haar wavelet transform for ternary-valued signals {-1, 0, +1}.

[![crates.io](https://img.shields.io/crates/v/ternary-haar.svg)](https://crates.io/crates/ternary-haar)

## Overview

The Haar wavelet is the simplest possible wavelet — a step function that decomposes
signals into approximation (low-frequency) and detail (high-frequency) components
at multiple resolution levels. It is ideally suited for ternary signals because
piecewise-constant signals over {-1, 0, +1} produce sparse wavelet representations.

This crate provides:

- **Haar wavelet decomposition** — Multi-resolution analysis with O(N log N) complexity
- **Perfect reconstruction** — Inverse transform recovers the original signal exactly
- **Wavelet coefficients** — Flat coefficient vector in standard ordering
- **Coefficient thresholding** — Lossy compression by discarding small coefficients
- **Compression metrics** — Compression ratio, reconstruction error (MSE), energy preservation
- **Sparsity analysis** — Count zero/near-zero coefficients for compression planning

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ternary-haar = "0.1.0"
```

## Quick Start

```rust
use ternary_haar::*;

// Ternary signal (length must be a power of 2)
let signal = vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];

// Full multi-resolution decomposition
let decomp = decompose(&signal);

// Reconstruct (lossless)
let recovered = reconstruct(&decomp);

// Get flat wavelet coefficients
let coeffs = wavelet_coefficients(&decomp);

// Compress by thresholding small coefficients
let compressed = threshold_compress(&decomp, 0.1);
let compressed_signal = reconstruct(&compressed);

// Measure quality
let error = reconstruction_error(&signal, &compressed_signal);
let energy = energy_preserved(&signal, &decomp, 0.1);
let (non_zero, total, ratio) = compression_ratio(&decomp, 0.1);
```

## Mathematical Background

### Haar Wavelet

The Haar wavelet ψ(t) and scaling function φ(t) are defined as:

```
ψ(t) = { +1  for 0 ≤ t < 1/2
        { -1  for 1/2 ≤ t < 1
        {  0  otherwise

φ(t) = { 1  for 0 ≤ t < 1
       { 0  otherwise
```

### Single-Level Transform

For a discrete signal x of length N, one level of the Haar transform produces:

```
Approximation: a[k] = (x[2k] + x[2k+1]) / √2
Detail:        d[k] = (x[2k] - x[2k+1]) / √2
```

### Multi-Resolution Analysis

The decomposition is applied recursively to the approximation coefficients:

```
Level 1: x → (a₁, d₁)          where a₁ has length N/2
Level 2: a₁ → (a₂, d₂)         where a₂ has length N/4
Level 3: a₂ → (a₃, d₂)         where a₃ has length N/8
...
Until a single approximation coefficient remains
```

### Why Haar for Ternary?

Ternary signals {-1, 0, +1} are piecewise-constant. The Haar wavelet is optimal for
such signals because:

1. **Sparsity**: Constant regions produce zero detail coefficients
2. **Edges detected**: Transitions between {-1, 0, +1} produce large detail coefficients
3. **Energy compaction**: Most energy concentrates in few coefficients
4. **Fast computation**: O(N) per level, O(N log N) total

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `HaarLevel` | One decomposition level with `approximation` and `detail` vectors |
| `HaarDecomposition` | Full multi-level decomposition with `levels` and `original_len` |

### Functions

| Function | Description |
|----------|-------------|
| `decompose(signal)` | Full multi-resolution Haar decomposition |
| `reconstruct(decomp)` | Perfect reconstruction from decomposition |
| `wavelet_coefficients(decomp)` | Flat coefficient vector [d₁, d₂, ..., dₙ, aₙ] |
| `threshold_compress(decomp, threshold)` | Zero out small detail coefficients |
| `compression_ratio(decomp, threshold)` | Compute (non_zero, total, ratio) |
| `reconstruction_error(original, reconstructed)` | Mean squared error |
| `signal_energy(signal)` | Sum of squared values |
| `energy_preserved(original, decomp, threshold)` | Fraction of energy after compression |
| `count_zero_coefficients(decomp, threshold)` | Count near-zero coefficients |

## Properties Verified by Tests

- **Perfect reconstruction**: decompose → reconstruct recovers the original signal exactly
- **Coefficient count**: Total wavelet coefficients always equal signal length
- **Energy conservation**: Wavelet coefficient energy equals signal energy (Parseval)
- **Constant signal sparsity**: All detail coefficients are zero for constant input
- **Piecewise-constant sparsity**: Step functions produce sparse representations
- **Zero signal**: Produces all-zero coefficients at every level
- **Threshold monotonicity**: Higher thresholds preserve less energy
- **Lossless at zero threshold**: No error when threshold is 0

## License

MIT
