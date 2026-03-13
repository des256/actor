# THAT DOCKERFILE

`./build-genmei.sh` for the desktop
`./build-murdock.sh` for the jetson

But...

Make sure that the exact NVIDIA TensorRT tar file is available from `docker/`:

genmei: TensorRT-10.7.0.23.Linux.x86_64-gnu.cuda-12.6.tar.gz
murdock: TensorRT-10.7.0.23.l4t.aarch64-gnu.cuda-12.6.tar.gz

## Details

Linux: Ubuntu 22.04
CUDA: 12.6
TensorRT: 10.7.0.23
TensorRT-LLM:

## When Ready...

They're called `actor:genmei` and `actor:murdock`. Used to build engines, run tools and run the apps as well.
