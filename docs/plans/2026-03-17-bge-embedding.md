# BGE Text Embedding Implementation Plan

Created: 2026-03-17
Status: VERIFIED
Approved: Yes
Iterations: 0
Worktree: No
Type: Feature

## Summary

**Goal:** Add BGE-small-en-v1.5 text embedding support — download the model, export to optimized ONNX, and implement CPU inference in Rust.
**Architecture:** Single `build_onnx.sh` script downloads the HuggingFace model and exports it to an optimized ONNX file. Rust implementation uses the existing `onnx` module (same pattern as `vad.rs`) for CPU inference with the `tokenizers` crate for WordPiece tokenization.
**Tech Stack:** Python (transformers, onnx, optimum), Rust (onnx module, tokenizers crate), ONNX Runtime CPU

## Scope

### In Scope

- Download `BAAI/bge-small-en-v1.5` via `huggingface-cli`
- Python export script to convert the model to optimized ONNX
- `build_onnx.sh` shell script orchestrating download + export
- Rust `bge.rs` implementation for CPU inference with L2-normalized embeddings
- Register `bge` module in `lib.rs`

### Out of Scope

- TensorRT / GPU inference (CPU ONNX chosen to avoid GPU contention)
- Batch embedding API (single sentence only, matching scaffolding)
- Separate `build_ckpt.sh` / `build_engine.sh` scripts

## Context for Implementer

> Write for an implementer who has never seen the codebase.

- **Patterns to follow:**
  - `actor/src/vad.rs:8-47` — canonical ONNX CPU inference pattern: create session, build input tensors, call `session.run()`, extract output
  - `actor/src/slm.rs:69-91` — ONNX session + tokenizer initialization pattern
  - `data/moonshine/build_ckpt.sh` — shell script structure (set -euo pipefail, validation, temp dir)
  - `data/moonshine/export.py` — ONNX export pattern (torch.onnx.export with dynamic_axes)

- **Conventions:**
  - Engine/model paths are `const` strings at top of module
  - ONNX sessions created via `onnx.create_session(executor, optimization_level, threads, path)`
  - Tensors created via `onnx::Value::from_slice()`, extracted via `.extract_tensor::<f32>()`
  - Modules declared as `pub mod bge;` in `lib.rs`

- **Key files:**
  - `actor/src/onnx/mod.rs` — ONNX Runtime FFI wrapper (Session, Value, Onnx types)
  - `actor/src/vad.rs` — simplest ONNX inference example
  - `actor/src/bge.rs` — existing scaffolding (will be rewritten)

- **Gotchas:**
  - The existing `bge.rs` scaffolding uses TensorRT — must be fully rewritten for ONNX Runtime
  - `EMBEDDING_SIZE` in scaffolding says 1024 — BGE-small-en-v1.5 is actually 384
  - Scaffolding has typo `Enbedding` → `Embedding`
  - Scaffolding references undefined `ENCODER_PATH` constant
  - BGE-small-en-v1.5 uses WordPiece tokenizer (`tokenizer.json`), not SentencePiece

- **Domain context:**
  - BGE-small-en-v1.5 is a 33M-parameter BERT-based encoder (6 layers, 12 heads, 384 hidden dim)
  - Input: `input_ids` + `attention_mask` + `token_type_ids` (all int64)
  - Output: `last_hidden_state` [batch, seq_len, 384] — take CLS token (index 0), L2-normalize
  - Max sequence length: 128 tokens (user choice)
  - No instruction prefix needed for this model variant (unlike larger BGE models)

## Assumptions

- `huggingface-cli` is available in PATH — supported by finding in `data/moonshine/` and `data/pocket/` patterns. Tasks 1 depend on this.
- `transformers` and `onnx` Python packages are available — supported by existing export scripts. Task 1 depends on this.
- The `tokenizers` crate (already in Cargo.toml) can load BGE's `tokenizer.json` — supported by HuggingFace tokenizers compatibility. Task 2 depends on this.
- BGE-small-en-v1.5 ONNX export produces standard BERT inputs (`input_ids`, `attention_mask`, `token_type_ids`) — supported by model architecture. Tasks 1, 2 depend on this.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ONNX export produces unexpected input/output names | Low | Medium | Validate tensor names in export.py with assertions; print model IO info |
| tokenizer.json format incompatible with `tokenizers` crate | Low | High | BGE uses standard HuggingFace WordPiece tokenizer; same crate used for moonshine/slm |
| ONNX Runtime CPU performance insufficient | Low | Low | BGE-small is 33M params — CPU inference should be <10ms per sentence |

## Goal Verification

### Truths

1. `data/bge/onnx/model.onnx` exists and is a valid ONNX model after running `build_onnx.sh`
2. `data/bge/onnx/tokenizer.json` exists after running `build_onnx.sh`
3. `Bge::new()` loads the ONNX session and tokenizer without panicking
4. `Bge::embed("hello world")` returns a 384-dimensional `Embedding` vector
5. Embedding vectors are L2-normalized (magnitude ~1.0)
6. The crate compiles with `bge` module declared in `lib.rs`

### Artifacts

1. `data/bge/build_onnx.sh` — build script
2. `data/bge/export.py` — ONNX export script
3. `data/bge/onnx/model.onnx` — exported model (generated)
4. `data/bge/onnx/tokenizer.json` — tokenizer (generated)
5. `actor/src/bge.rs` — Rust embedding implementation
6. `actor/src/lib.rs` — module declaration

## Progress Tracking

- [x] Task 1: Build script and ONNX export
- [x] Task 2: Rust BGE implementation
      **Total Tasks:** 2 | **Completed:** 2 | **Remaining:** 0

## Implementation Tasks

### Task 1: Build script and ONNX export

**Objective:** Create `data/bge/build_onnx.sh` that downloads the model and exports it to optimized ONNX format in `data/bge/onnx/`.

**Dependencies:** None

**Files:**

- Create: `data/bge/export.py`
- Create: `data/bge/build_onnx.sh`

**Key Decisions / Notes:**

- `build_onnx.sh` calls `huggingface-cli download BAAI/bge-small-en-v1.5 --local-dir source/` to fetch the model
- `export.py` loads the model via `transformers.AutoModel`, exports to ONNX with `torch.onnx.export`
- Dynamic axes on `input_ids`, `attention_mask`, `token_type_ids` for batch and sequence dimensions
- Use `onnx` + `onnxruntime` to optimize the model (graph optimization level ALL)
- Copy `tokenizer.json` from source to onnx/ directory
- Use opset 18 to match existing export scripts
- Script runs from `data/bge/` directory (same pattern as moonshine/pocket)

**Definition of Done:**

- [ ] `build_onnx.sh` downloads model to `source/` via huggingface-cli
- [ ] `export.py` exports valid ONNX model to `onnx/model.onnx`
- [ ] `onnx/tokenizer.json` copied from source
- [ ] Script is executable and uses `set -euo pipefail`

**Verify:**

- `cd data/bge && bash build_onnx.sh`
- `python3 -c "import onnx; m = onnx.load('data/bge/onnx/model.onnx'); print([i.name for i in m.graph.input])"`

### Task 2: Rust BGE implementation

**Objective:** Rewrite `actor/src/bge.rs` to use ONNX Runtime CPU for text embedding, and declare the module in `lib.rs`.

**Dependencies:** Task 1 (needs model files to test against)

**Files:**

- Modify: `actor/src/bge.rs` (full rewrite)
- Modify: `actor/src/lib.rs` (add `pub mod bge;`)

**Key Decisions / Notes:**

- Follow `vad.rs` pattern: struct holds `onnx::Session` + pre-allocated tensors
- Use `tokenizers::Tokenizer` for WordPiece tokenization (same crate as moonshine/slm)
- Model inputs: `input_ids` [1, seq_len] i64, `attention_mask` [1, seq_len] i64, `token_type_ids` [1, seq_len] i64
- Model output: `last_hidden_state` [1, seq_len, 384] f32
- Extract CLS token (first token, index 0), L2-normalize to produce final embedding
- `EMBEDDING_SIZE` = 384 (not 1024 as in scaffolding)
- `Embedding` struct wraps `Vec<f32>`, expose `as_slice()` for downstream use
- Constructor takes `&Arc<onnx::Onnx>` and `onnx::Executor` (use `Executor::Cpu`)
- `embed(&mut self, sentence: &str) -> Embedding` — synchronous, no async needed for CPU
- Signature change from scaffolding: `&self` → `&mut self` (input tensors are mutable), remove `async`, take `onnx::Onnx` instead of `tensorrt::Tensorrt`

**Definition of Done:**

- [ ] `bge.rs` compiles with ONNX Runtime CPU inference
- [ ] `lib.rs` declares `pub mod bge;`
- [ ] `Bge::new()` loads session and tokenizer
- [ ] `Bge::embed()` tokenizes input, runs inference, returns L2-normalized 384-dim embedding
- [ ] `Embedding::as_slice()` returns a `&[f32]` view of the embedding vector
- [ ] L2 normalization uses epsilon guard against zero-magnitude vectors
- [ ] Crate compiles cleanly (`cargo check`)

**Verify:**

- `cd /home/desmond/actor && cargo check`
