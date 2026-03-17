docker run -it --rm --gpus all --user $(id -u):$(id -g) -e HOME=/tmp -e CARGO_HOME=/tmp/.cargo -v .:/actor -w /actor actor:latest cargo run --release --bin $1
