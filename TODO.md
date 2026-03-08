- get parakeet TDT v3 maybe, because speaker embedding and multilingual
- SetFit (like all-MiniLM-L6-v2) or FastText for intent classification
- keyword/heuristic, when user sentence starts with 'hey' or 'hello', you can immediately classify it as a greeting
- DistilRoBERTa-base-emotion ran on the first 5-8 words of the LLM output to find nuance
- or, make Llama 3.2 output the nuance by itself
- Use TensorRT-LLM for Llama 3.2 --> 5x speed increase, but
  limit context, because it takes too much memory
- history summarizer, but only call it in spare time
- high level steering system
- proactive turn taking (robot speaks first)
- XML-style inline tagged animation
- audio-to-viseme
- speaker diarization
- implement VLM as main LLM
- Qwen3 2.5 for main LLM (try)
- clean up st7789.rs

"After days in the shade, sunlight now cuts through the rough surface, sending shimmering rays dancing across the rocky bed of the river, and illuminating the patches of bright-green algae that carpeted the rocks of deeper, slower pools."

## Optimize for Jetson

### Keep as ONNX

1. get the model from original source (safetensors or whatever)
2. convert to ONNX q4f16
3. in the code: use ONNX bindings in Rust

### Build TensorRT engine

1. get the model from original source (safetensors or whatever)
2. create TensorRT checkpoint in q4f16
3. on the Jetson: build the optimized engine
4. in the code: use simple FFI access from Rust

### Build TensorRT-LLM

1. get the model from original source (safetensors or whatever)
2. create TensorRT-LLM checkpoint in q4f16
3. on the Jetson: run trtllm-build to build the optimized engine
4. in the code: use simple FFI access from Rust
