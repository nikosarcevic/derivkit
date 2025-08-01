Introduction
============

**DerivKit** is a robust Python toolkit for computing numerical derivatives with high stability and precision.

It implements two core methods:

- **STEM** — an adaptive algorithm that improves derivative estimates for noisy or nonanalytic functions,
- **Five-point stencil** — a classic finite-difference method for benchmarking.

This package is ideal for scientific computing tasks such as:

- Fisher matrix forecasting,
- Numerical likelihood evaluations,
- Cosmological parameter estimation,
- Any application where derivative stability is critical.

**Why use DerivKit?**

- **Noise-resistant**: STEM handles unstable or irregular functions gracefully.
- **Fast and adaptive**: Efficient even with expensive function calls.
- **Modular and extensible**: Works seamlessly with NumPy and can be dropped into larger pipelines.
- **Benchmark-ready**: Includes visualization and comparison tools.

**DerivKit** is general-purpose and lightweight — although built with cosmology in mind,
it's useful in any domain where high-quality numerical derivatives are needed.
