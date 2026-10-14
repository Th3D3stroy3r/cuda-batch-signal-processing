# CUDA Multi-Stream Batch Signal Denoising Pipeline

## Overview
This project implements a high-throughput GPU signal processing pipeline using CUDA C++ (following the Google C++ Style Guide). In a single batch execution, it generates and processes **200 time-series CSV signal files** (each containing 1,024 floating-point samples corrupted by high-frequency Gaussian noise, totaling **204,800 data points**) using 4 concurrent CUDA streams and an asynchronous 1D Moving-Average Low-Pass Filter CUDA kernel.

## Repository Structure
- `src/signal_processor.h`: Header file defining configuration structures, CUDA error macros, and function declarations.
- `src/signal_processor.cu`: Main CUDA implementation containing the `MovingAverageDenoiseKernel`, multi-stream asynchronous memory manager, and CLI parser.
- `Makefile`: Build configuration for compiling `signal_processor.exe` with `nvcc`.
- `run.sh`: Automated execution script that compiles the binary, runs the 200-file batch job with CLI flags, and saves the execution trace.
- `data/input/`: Contains the 200 raw noisy input CSV signal files (`signal_000.csv` to `signal_199.csv`).
- `data/output/`: Contains the 200 GPU-filtered output CSV signal files (`denoised_signal_000.csv` to `denoised_signal_199.csv`).
- `logs/execution_proof.log`: Complete console output log verifying batch GPU execution across all 200 files.

## How to Build and Run

### Quick Start (Automated Script)
```bash
./run.sh
```

### Using Make
```bash
make clean
make build
make run
```

### Command Line Interface (CLI) Arguments
You can customize execution parameters directly via the CLI:
```bash
./signal_processor.exe \
  --input_dir data/input \
  --output_dir data/output \
  --num_files 200 \
  --signal_length 1024 \
  --filter_window 7
```
