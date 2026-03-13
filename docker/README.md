# PLAN

## Base

- Ubuntu 22
- Python 3.10

## CUDA + TensorRT

- Whichever CUDA is available, should be >= 12.6
- TensorRT 10.3, based on CUDA 12.6

## TensorRT-LLM

- build TensorRT-LLM for CUDA 12.6, TensorRT 10.3, specifically for architectures 87 (Jetson) and 89 (Desktop)

# USAGE

- build engines
- run tests
- run experiments
