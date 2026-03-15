cp /usr/lib/aarch64-linux-gnu/libnvdla_compiler.so libnvdla_compiler.so
cp /usr/lib/aarch64-linux-gnu/libnvcudla.so libnvcudla.so
docker build -f Dockerfile.murdock -t actor .
rm libnvdla_compiler.so libnvcudla.so
