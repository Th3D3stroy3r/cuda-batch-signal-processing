#!/usr/bin/env bash
set -e

mkdir -p data/input data/output logs

echo "Building CUDA Signal Processor..."
make build

echo "Executing Batch GPU Signal Processing on 200 CSV signal files..."
./signal_processor.exe \
  --input_dir data/input \
  --output_dir data/output \
  --num_files 200 \
  --signal_length 1024 \
  --filter_window 7 | tee logs/execution_proof.log

echo "Execution log saved to logs/execution_proof.log"
