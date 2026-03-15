Each model has a `source` folder with the originally downloaded model.

`build_ckpt.sh` creates a `ckpt` folder with checkpoint ONNX files for further processing. Once the checkpoint exists, the source can be removed/ignored.

On the target platform (desktop or jetson), run `build_engine.sh` which generates the TensorRT(-LLM) engines in the `engine` folder. These are loaded by the inference code.
