# Llama3 TensorRT Migration Plan

Created: 2026-03-17
Status: VERIFIED
Approved: Yes
Iterations: 0
Worktree: No
Type: Feature

## Summary

**Goal:** Replace TensorRT-LLM based Llama3 inference with plain TensorRT inference, eliminating the Docker/TRT-LLM dependency entirely. Create a custom ONNX export script and trtexec-based engine builder. Remove all TRT-LLM code.

**Architecture:** The new Llama3 module will use plain TensorRT engines (like Pocket TTS) with manual KV cache management on GPU. A custom `export.py` will convert HuggingFace Llama3 safetensors to a single ONNX model with explicit KV cache I/O tensors. `trtexec` builds the engine with FP16 precision (quantization deferred). The Rust module keeps the existing Handle/Listener/create channel-based streaming pattern.

**Tech Stack:** Python (torch, transformers, onnx) for export, trtexec for engine building, Rust with TensorRT C FFI for inference.

## Scope

### In Scope

- Custom ONNX export script (`data/llama3/export.py`) for Llama 3.2 3B with explicit KV cache I/O
- TensorRT engine build script (`data/llama3/build_engine.sh`) using trtexec with FP16
- Rewrite `actor/src/llama3.rs` to use plain TensorRT (GPU buffers, KV cache, greedy decode)
- Update `actor/src/bin/test_llama3.rs` — remove `trtllm_enabled` gates
- Remove all TRT-LLM code: `executor.rs`, TRT-LLM sections from `ffi.rs`, `ffi.cpp`, `build.rs`, `mod.rs`, `lib.rs`
- Remove `data/llama3/build_ckpt.sh` (TRT-LLM checkpoint step no longer needed)

### Out of Scope

- W4A8 or INT8 quantization (FP16 first, quantization added later)
- Modifying `slm.rs` (ONNX-based SLM module stays as-is)
- Modifying `chat.rs` (it uses `slm::`, not `llama3::`)
- Docker image changes (Dockerfiles kept as-is for now)
- Jetson aarch64 build (same approach works, just different trtexec flags)

## Context for Implementer

> Write for an implementer who has never seen the codebase.

- **Patterns to follow:**
  - `actor/src/pocket/trt.rs` — the reference for TRT engine + KV cache + GPU buffers in this codebase. It uses `tensorrt::Tensorrt`, `tensorrt::Buffer`, `tensorrt::Context` with manual `set_input_shape` / `set_tensor_address` / `enqueue` / `cudaStreamSynchronize`.
  - `actor/src/slm.rs` — the reference for ONNX-based LLM inference with KV cache. Shows the autoregressive decode loop, token sampling, incremental decoding with the `tokenizers` crate. The new `llama3.rs` replicates this logic but on TRT engines instead of ONNX sessions.
  - `data/pocket/export.py` — reference for ONNX export with explicit KV cache tensors as inputs/outputs.
  - `data/pocket/build_engine.sh` — reference for trtexec engine building with dynamic shapes.

- **Conventions:**
  - Compact imports: single nested `use` block per file.
  - Re-export pattern: `mod x; pub use x::*;`
  - `size_of::<f32>()` for byte calculations.
  - GPU buffers are pre-allocated at max capacity, dynamic shapes set per-inference via `set_input_shape`.
  - CUDA stream created once, synchronized after each `enqueue`.

- **Key files:**
  - `actor/src/tensorrt/mod.rs` — TensorRT module root, re-exports Buffer, Context, Engine, Tensorrt
  - `actor/src/tensorrt/buffer.rs` — GPU buffer with upload/download (cudaMalloc/cudaMemcpy)
  - `actor/src/tensorrt/context.rs` — TRT execution context wrapper
  - `actor/src/tensorrt/engine.rs` — TRT engine wrapper with `get_io_tensors()`, `create_context()`
  - `actor/src/tensorrt/tensorrt.rs` — TRT runtime wrapper with `load_engine()`
  - `actor/build.rs` — links ONNX, CUDA, TensorRT, and (currently) TRT-LLM

- **Gotchas:**
  - The `_GLIBCXX_USE_CXX11_ABI=0` define in `build.rs` is required for TensorRT compatibility.
  - KV cache uses double-buffering (A/B buffers, swapped each step). TensorRT does not support aliasing input and output tensor addresses within a single enqueue — separate buffers are required. Llama3 has 28 layers × 8 KV heads × 128 head_dim = 56 buffer pairs (112 buffers total). At max_seq_len=2048 FP16, each KV buffer is ~4MB, total KV memory ~448MB. This is acceptable for desktop GPUs but tight on Jetson (deferred).
  - `tokenizers` crate is already a dependency — used for both `llama3.rs` and `slm.rs`.

- **Domain context:**
  - Llama 3.2 3B config: 28 layers, 24 attention heads, 8 KV heads (GQA), hidden_size=3072, intermediate_size=8192, head_dim=128, vocab_size=128256, RoPE with llama3-style scaling.
  - The ONNX export must handle: embedding lookup, RMSNorm, RoPE, GQA attention, SwiGLU FFN, and lm_head projection.
  - KV cache shape per layer: `[batch=1, num_kv_heads=8, seq_len, head_dim=128]`. Total 28 layers × 2 (K+V).

## Assumptions

- Llama 3.2 3B Instruct safetensors are already available in `data/llama3/source/` — supported by existing files in that directory. Tasks 1-3 depend on this.
- The host has CUDA toolkit + TensorRT installed natively (or accessible via paths in `build.rs`) — supported by existing `build.rs` search paths. Tasks 2-4 depend on this.
- A single ONNX model with 56 KV I/O tensors (28 layers × 2) will work with trtexec — Pocket TTS uses the same pattern at smaller scale (12 tensors). trtexec shape flags are generated programmatically in the build script to avoid command-line length limits. Tasks 1-2 depend on this.
- FP16 precision is sufficient for initial validation — user confirmed. Task 2 depends on this.
- The existing Handle/Listener channel pattern is sufficient for streaming — user confirmed. Task 3 depends on this.
- TensorRT requires separate input/output buffers for KV cache — TRT does not support aliasing input and output tensor addresses to the same buffer within a single enqueue call. Double-buffering (A/B swap) is used, matching Pocket TTS. Task 3 depends on this.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ONNX export fails for Llama3's RoPE scaling | Medium | High | Fall back to simpler RoPE (standard, not llama3-style) in export; add custom RoPE op if needed |
| trtexec rejects dynamic KV cache shapes | Low | High | Pocket TTS already uses this pattern successfully; match its shape specification format |
| KV cache memory exceeds GPU for long contexts | Medium | Medium | Cap max_seq_len to 2048 in engine build (matching current TRT-LLM config); this fits in ~1.7GB at FP16 |
| Greedy decode quality differs from TRT-LLM | Low | Low | Greedy decode is deterministic; verify output matches expected Llama3 behavior |
| trtexec rejects 112 shape specification args for 56 KV tensors | Low | High | Generate shape flags programmatically in build_engine.sh; if command-line too long, use trtexec timing cache or split into profile JSON file |
| ONNX export produces semantically broken model (wrong attention, RoPE, or FFN logic) | Medium | High | Validate ONNX output vs PyTorch reference: run both on same input, compare logits within tolerance 1e-3. Add numerical validation to Task 1 |

## Goal Verification

### Truths

1. `test_llama3` binary compiles and runs without Docker, without `trtllm_enabled` flag
2. `test_llama3` produces coherent text output for a given prompt
3. `cargo build --release` succeeds without TRT-LLM libraries installed
4. `data/llama3/export.py` produces a valid ONNX model from source safetensors
5. `data/llama3/build_engine.sh` produces a TensorRT engine from the ONNX model using trtexec (not trtllm-build)
6. No `trtllm_enabled` cfg flag exists anywhere in the codebase
7. No `executor.rs` file exists in `actor/src/tensorrt/`

### Artifacts

1. `actor/src/llama3.rs` — rewritten to use plain TRT
2. `actor/src/bin/test_llama3.rs` — no cfg gates
3. `data/llama3/export.py` — custom ONNX export script
4. `data/llama3/build_engine.sh` — trtexec-based engine builder
5. `actor/build.rs` — TRT-LLM sections removed
6. `actor/src/tensorrt/ffi.rs`, `ffi.cpp`, `mod.rs` — TRT-LLM code removed

## Progress Tracking

- [x] Task 1: Create ONNX export script
- [x] Task 2: Create TensorRT engine build script
- [x] Task 3: Rewrite llama3.rs for plain TensorRT
- [x] Task 4: Update test_llama3.rs
- [x] Task 5: Remove all TRT-LLM code
      **Total Tasks:** 5 | **Completed:** 5 | **Remaining:** 0

## Implementation Tasks

### Task 1: Create ONNX Export Script

**Objective:** Write `data/llama3/export.py` that converts Llama 3.2 3B Instruct safetensors to a single ONNX model with explicit KV cache I/O tensors for each of the 28 layers.

**Dependencies:** None

**Files:**

- Create: `data/llama3/export.py` (replace current TRT-LLM converter)
- Delete: `data/llama3/build_ckpt.sh` (no longer needed)

**Key Decisions / Notes:**

- Follow the Pocket TTS export pattern (`data/pocket/export.py`) but for a decoder-only LLM.
- The ONNX model must have these I/O tensors:
  - **Inputs:** `input_ids` [1, seq_len], `position_ids` [1, seq_len], 28× `past_key_values.N.key` [1, 8, past_len, 128], 28× `past_key_values.N.value` [1, 8, past_len, 128]
  - **Outputs:** `logits` [1, seq_len, 128256], 28× `present.N.key` [1, 8, new_len, 128], 28× `present.N.value` [1, 8, new_len, 128]
- Use `torch.onnx.export` with `opset_version=18` and proper `dynamic_axes` for seq_len and past_len.
- The wrapper module must implement: token embedding, RMSNorm, RoPE (llama3-style with scaling), GQA attention with KV cache concat, SwiGLU FFN, lm_head.
- Export in float32, let trtexec handle FP16 conversion.
- Use model config from `data/llama3/source/config.json` to read architecture params.

**Definition of Done:**

- [ ] `export.py` runs successfully: `python3 export.py source/ onnx/`
- [ ] Produces `onnx/model.onnx` with correct I/O tensor names and shapes
- [ ] ONNX model has exactly 58 inputs (input_ids + position_ids + 56 KV tensors) and 57 outputs (logits + 56 KV tensors)
- [ ] ONNX model passes `onnx.checker.check_model()`
- [ ] Numerical validation: compare ONNX inference output vs PyTorch forward for a short test sequence (logits match within 1e-3 tolerance)
- [ ] No TRT-LLM imports in the script

**Verify:**

- `cd data/llama3 && python3 export.py source/ onnx/`

---

### Task 2: Create TensorRT Engine Build Script

**Objective:** Write `data/llama3/build_engine.sh` that builds a TensorRT engine from the ONNX model using trtexec with FP16 precision and dynamic shapes for KV cache.

**Dependencies:** Task 1

**Files:**

- Modify: `data/llama3/build_engine.sh` (replace trtllm-build with trtexec)

**Key Decisions / Notes:**

- Follow the Pocket TTS pattern (`data/pocket/build_engine.sh`).
- Use `trtexec` directly (NOT via Docker — the whole point of this migration). If the user's host has trtexec on PATH, use it directly. If not, provide a Docker fallback line (commented out).
- Dynamic shapes:
  - `input_ids`: min [1,1], opt [1,128], max [1,2048]
  - `position_ids`: min [1,1], opt [1,128], max [1,2048]
  - Each `past_key_values.N.key`: min [1,8,0,128], opt [1,8,1024,128], max [1,8,2048,128]
  - Each `past_key_values.N.value`: same as key
- Flags: `--fp16 --builderOptimizationLevel=5 --memPoolSize=workspace:4096`
- Copy tokenizer.json to engine directory.

**Definition of Done:**

- [ ] `build_engine.sh` completes without errors (handles 112 shape flags for 56 KV tensors)
- [ ] Produces `engine/model.engine` file
- [ ] Uses trtexec (not trtllm-build)
- [ ] No Docker dependency in the default path
- [ ] Shape flags generated programmatically (not hand-written) to avoid errors

**Verify:**

- `cd data/llama3 && bash build_engine.sh`

---

### Task 3: Rewrite llama3.rs for Plain TensorRT

**Objective:** Replace the TRT-LLM Executor-based inference in `llama3.rs` with plain TensorRT engine execution using GPU buffers and manual KV cache management, following the Pocket TTS pattern.

**Dependencies:** Task 1 (ONNX tensor names), Task 2 (engine path)

**Files:**

- Modify: `actor/src/llama3.rs`

**Key Decisions / Notes:**

- Keep the same public API: `Input<T>`, `Output<T>`, `Handle<T>`, `Listener<T>`, `create<T>()`.
- Add a `TrtLlama3` struct (similar to `TrtPocket` in `pocket/trt.rs`) that holds:
  - `Arc<tensorrt::Tensorrt>` (shared runtime)
  - `Arc<tensorrt::Context>` (execution context)
  - CUDA stream
  - GPU `Buffer` for each I/O tensor: input_ids, position_ids, logits, 28× KV pairs (in + out)
  - `Tokenizer` from the `tokenizers` crate
- KV cache approach: Pre-allocate buffers at max_seq_len=2048. Each step:
  1. Upload input_ids [1,1] and position_ids [1,1] to GPU
  2. Set input shapes for dynamic dims (seq_len=1, past_len=current)
  3. Bind past KV buffers as inputs, present KV buffers as outputs
  4. Enqueue + synchronize
  5. Download logits, argmax for next token
  6. Swap past/present buffers (double-buffer like Pocket TTS)
- For the prefill phase (initial prompt), process all tokens at once (seq_len=prompt_len).
- Greedy decode (argmax). Sampling can be added later.
- Use incremental token decoding (like `slm.rs` lines 283-303) for streaming.
- Architecture constants from config: NUM_LAYERS=28, NUM_KV_HEADS=8, HEAD_DIM=128, VOCAB_SIZE=128256, MAX_SEQ_LEN=2048.
- Engine path: `data/llama3/engine/model.engine`, tokenizer: `data/llama3/engine/tokenizer.json`.

**Definition of Done:**

- [ ] `llama3.rs` compiles without `trtllm_enabled` flag
- [ ] `TrtLlama3::new()` loads engine and allocates GPU buffers
- [ ] Autoregressive generation loop produces token stream via channels
- [ ] No references to `tensorrt::Executor` or TRT-LLM types
- [ ] Handles EOS/EOT token detection and max_tokens limit
- [ ] Runtime smoke test: engine loads, prefill runs without CUDA errors, single decode step produces valid vocab ID (0-128255)

**Verify:**

- `cargo build --release --bin test_llama3`
- `cargo run --release --bin test_llama3` (requires engine file — run after Tasks 1+2)

---

### Task 4: Update test_llama3.rs

**Objective:** Remove all `#[cfg(trtllm_enabled)]` gates from `test_llama3.rs` so it compiles and runs unconditionally, using the new plain TRT API.

**Dependencies:** Task 3

**Files:**

- Modify: `actor/src/bin/test_llama3.rs`

**Key Decisions / Notes:**

- Remove both `#[cfg(not(trtllm_enabled))]` and `#[cfg(trtllm_enabled)]` blocks.
- The `llama3::create::<()>(&epoch)` call stays the same since the public API doesn't change.
- Need to update the `Direct` usage if the constructor now takes a `Tensorrt` Arc. Adjust the `create` function signature in llama3.rs to accept a `&Arc<tensorrt::Tensorrt>` parameter (matching Pocket TTS's pattern where `create` takes `&trt`).

**Definition of Done:**

- [ ] No `trtllm_enabled` references in `test_llama3.rs`
- [ ] Binary compiles: `cargo build --release --bin test_llama3`
- [ ] Test structure preserved: warmup run + measured run with TTFT

**Verify:**

- `cargo build --release --bin test_llama3`

---

### Task 5: Remove All TRT-LLM Code

**Objective:** Clean removal of all TRT-LLM code, cfg flags, and Docker build dependencies from the codebase.

**Dependencies:** Tasks 3, 4

**Files:**

- Delete: `actor/src/tensorrt/executor.rs`
- Modify: `actor/src/tensorrt/ffi.rs` — remove all `#[cfg(trtllm_enabled)]` blocks (TrtLlmModelType, TrtLlmExecutor, all trtllm_* function declarations)
- Modify: `actor/src/tensorrt/ffi.cpp` — remove `#ifdef TRTLLM_ENABLED` section (TRT-LLM includes, TrtLlmExecutor struct, all trtllm_* functions)
- Modify: `actor/src/tensorrt/mod.rs` — remove `#[cfg(trtllm_enabled)] mod executor;` and its re-export
- Modify: `actor/src/lib.rs` — change `#[cfg(trtllm_enabled)] pub mod llama3;` to unconditional `pub mod llama3;`
- Modify: `actor/build.rs` — remove TRT-LLM detection (`trtllm_include`, `trtllm_enabled` flag, TRT-LLM link directives, `trtllm_include` in cc::Build), remove `cargo::rustc-check-cfg=cfg(trtllm_enabled)` line

**Key Decisions / Notes:**

- **Ordering:** By the time Task 5 runs, Tasks 3+4 have already rewritten llama3.rs and test_llama3.rs to use plain TRT — the TRT-LLM code is dead/unreachable. Run `cargo run --release --bin test_llama3` BEFORE starting Task 5 to confirm the new implementation works. If it fails, debug Tasks 3/4 while the old code is still present as reference.
- Also remove `ffi.h` if it exists and contains TRT-LLM declarations (check during implementation).
- The `cc::Build` for `ffi.cpp` stays — it still compiles the plain TensorRT C-to-C++ bindings.
- `build.rs` keeps: ONNX, CUDA, and plain TensorRT linking.
- Verify no remaining references to `trtllm` anywhere via grep.

**Definition of Done:**

- [ ] `grep -r trtllm actor/` returns zero results
- [ ] `grep -r TRTLLM actor/` returns zero results
- [ ] `cargo build --release` succeeds without TRT-LLM installed
- [ ] `executor.rs` does not exist
- [ ] `build.rs` has no TRT-LLM detection or linking

**Verify:**

- `cargo build --release`
- `grep -rn "trtllm\|TRTLLM\|trt_llm\|TRT.LLM" actor/`

## Open Questions

None — all design decisions resolved.

## Deferred Ideas

- W4A8 quantization via GPTQ/AWQ pre-quantized ONNX export
- INT8 post-training quantization with trtexec calibration
- Batched inference support (currently single-request only)
- Jetson-specific engine build flags (SM87 compute capability)
