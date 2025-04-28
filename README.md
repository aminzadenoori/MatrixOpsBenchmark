# MatrixOpsBenchmark

This repository contains a Python script to benchmark matrix operations for statistical analysis across different platforms (NVIDIA GPUs, Intel CPUs, Apple M3), precisions (F32, F64), and matrix types (sparse, non-sparse). It generates performance plots similar to those used in computational studies, comparing execution times for various operations over a range of matrix sizes.

## Overview

The script performs the following matrix operations:
- Element-wise product
- Exponential of each element
- Means of the rows
- Matrix product
- Inverse of a matrix (non-sparse only)
- Singular Value Decomposition (SVD, non-sparse only)

**Key Features:**
- Matrix sizes: 1000x1000 to 5000x5000 (step 1000)
- Precisions: Single (F32) and double (F64)
- Matrix types: Sparse (1% density) and non-sparse
- Platforms: 
  - NVIDIA RTX 4080/L40S GPUs (Windows/Linux, using CuPy)
  - Intel CPUs (Windows/Linux, using NumPy with MKL)
  - Apple M3 (macOS, using NumPy with Accelerate)
- Reference: CPU F64 results are used as the accuracy reference
- Output: Log-scale plots of execution time vs. matrix size for each operation

## Prerequisites

### Software Requirements
- Python 3.8 or higher
- Operating Systems: Windows, Linux, or macOS

### Dependencies
Install the required Python packages using pip:
```bash
pip install numpy scipy matplotlib
```
