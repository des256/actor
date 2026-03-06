# XSLM Module Implementation Plan

Created: 2026-03-06
Status: VERIFIED
Approved: Yes
Iterations: 0
Worktree: No
Type: Feature

## Summary

**Goal:** Implement `src/xslm.rs` library module and `src/bin/xslm.rs` binary, mirroring the structure of the existing `src/slm.rs` and `src/bin/slm.rs`, to support ONNX-based SLM inference with SmolLM2 360M model (and future models).

**Architecture:** Direct copy of `slm.rs` pattern — model enum, channel-based I/O (`Input`/`Output`/`Handle`/`Listener`), `create()` factory function with ONNX session + tokenizer + threaded prompt pump. Prompt building added as a separate function in `prompt.rs`.

**Tech Stack:** Rust, ONNX Runtime (raw C FFI via `onnx` module), `tokenizers` crate, `tokio` channels

## Scope

### In Scope
- `actor/src/xslm.rs` — full module mirroring `slm.rs` structure with SmolLM2 360M constants
- `actor/src/bin/xslm.rs` — interactive chat binary mirroring `bin/slm.rs`
- `actor/src/prompt.rs` — new `build_xslm()` function for xslm models
- `actor/src/lib.rs` — register `pub mod xslm;`

### Out of Scope
- Additional xslm models beyond SmolLM2 360M (designed for future expansion but not populated)
- Changes to existing `slm.rs` or `bin/slm.rs`
- Tests (no unit tests exist for slm.rs; matching that pattern)

## Context for Implementer

> Write for an implementer who has never seen the codebase.

- **Patterns to follow:** `actor/src/slm.rs` — the entire file is the template. Copy structure exactly.
- **Conventions:**
  - Module uses `use { super::*, ... }` to import from lib.rs prelude
  - Model constants are file-level `const` values (path, tokenizer path, EOS tokens, num_kv_heads, head_dim)
  - Generic `<T: Clone + Send + 'static>` payload threading through Input/Output/Handle/Listener
  - Channels: `std::sync::mpsc` for input (sync sender), `tokio::sync::mpsc` for output (async receiver)
  - KV cache introspection from ONNX input names (`.key`/`.value` suffixes)
  - Greedy decoding (argmax of logits)
  - Incremental token decoding (track `generated_tokens` + `prev_decoded_len`)
- **Key files:**
  - `actor/src/slm.rs` — the source template (351 lines)
  - `actor/src/bin/slm.rs` — binary template (93 lines)
  - `actor/src/prompt.rs` — prompt building, needs new `build_xslm` function
  - `actor/src/lib.rs` — module registration
  - `actor/src/onnx/mod.rs` — ONNX Runtime FFI wrapper (Onnx, Session, Value, Executor types)
  - `actor/src/history.rs` — History/Role types used by prompt building and binary
  - `actor/src/epoch.rs` — Epoch type for staleness tracking
- **Gotchas:**
  - SmolLM2 360M does NOT use `cache_position` input (that's Gemma3-specific in slm.rs). Remove that special case.
  - The model file is `model_q4f16.onnx` (underscore), stored at `data/xslm/smollm2_360m/`
  - Panic messages should say "Xslm:" not "Slm:" to distinguish in logs
  - `prompt.rs` currently takes `slm::Model` — the new function takes `xslm::Model`
- **Domain context:** SmolLM2 360M uses ChatML prompt format (`<|im_start|>`/`<|im_end|>`) same as SmolLM3 in the existing slm module

## SmolLM2 360M Model Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Model path | `data/xslm/smollm2_360m/model_q4f16.onnx` | Filesystem |
| Tokenizer path | `data/xslm/smollm2_360m/tokenizer.json` | Filesystem |
| EOS tokens | `[0, 2]` (`<\|endoftext\|>`, `<\|im_end\|>`) | Tokenizer special tokens |
| num_kv_heads | 5 | Architecture (hidden_size=960, num_attn_heads=15) |
| head_dim | 64 | Architecture (960/15) |
| Prompt format | ChatML (`<\|im_start\|>`/`<\|im_end\|>`) | Same as SmolLM3 |

## Progress Tracking

- [x] Task 1: Create xslm library module + prompt extension
- [x] Task 2: Create xslm binary
**Total Tasks:** 2 | **Completed:** 2 | **Remaining:** 0

## Implementation Tasks

### Task 1: Create xslm library module + prompt extension

**Objective:** Create `actor/src/xslm.rs` mirroring `slm.rs` structure with SmolLM2 360M model, register it in `lib.rs`, and add `build_xslm` function to `prompt.rs`.

**Dependencies:** None

**Files:**
- Create: `actor/src/xslm.rs`
- Modify: `actor/src/lib.rs` (add `pub mod xslm;`)
- Modify: `actor/src/prompt.rs` (add `build_xslm` function)

**Key Decisions / Notes:**
- Copy `slm.rs` structure exactly: `Model` enum, `Input`/`Output`/`Handle`/`Listener` types, `create()` function
- Start with single variant `Model::Smollm2360m` (designed for future expansion via additional enum variants + constants)
- Remove `cache_position` handling (Gemma3-specific) since SmolLM2 doesn't use it
- Change all panic message prefixes from "Slm:" to "Xslm:"
- In `prompt.rs`, add `pub fn build_xslm(model: xslm::Model, ...)` with ChatML format (identical to the `Smollm3` branch in existing `build()`)

**Definition of Done:**
- [ ] `actor/src/xslm.rs` exists with `Model` enum, `Input`/`Output`/`Handle`/`Listener` types, and `create()` function
- [ ] `actor/src/lib.rs` registers `pub mod xslm;`
- [ ] `actor/src/prompt.rs` has `build_xslm` function
- [ ] `cargo check` passes with no errors

**Verify:**
- `cd /home/desmond/actor/actor && cargo check 2>&1`

### Task 2: Create xslm binary

**Objective:** Create `actor/src/bin/xslm.rs` mirroring `bin/slm.rs` — interactive chat binary using the xslm module.

**Dependencies:** Task 1

**Files:**
- Create: `actor/src/bin/xslm.rs`

**Key Decisions / Notes:**
- Mirror `bin/slm.rs` structure: model selection menu (single option for now), persona selection, chat loop
- Use `xslm::create::<()>()` instead of `slm::create::<()>()`
- Use `prompt::build_xslm()` instead of `prompt::build()`
- All xslm types: `xslm::Input`, `xslm::Output`, `xslm::Model`
- Model selection: show "1. SmolLM2 (360M)" as the only option (future models added as enum variants)

**Definition of Done:**
- [ ] `actor/src/bin/xslm.rs` exists with model selection, persona selection, and chat loop
- [ ] `cargo check --bin xslm` passes with no errors
- [ ] `cargo build --bin xslm` compiles successfully

**Verify:**
- `cd /home/desmond/actor/actor && cargo check --bin xslm 2>&1`

## Testing Strategy

- **Build verification:** `cargo check` and `cargo build --bin xslm`
- **Runtime verification:** Run `cargo run --bin xslm` to verify model loads and interactive chat works (requires CUDA GPU and ONNX runtime)
- **No unit tests:** Matching existing pattern — `slm.rs` has no unit tests

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SmolLM2 360M architecture params wrong (kv_heads, head_dim) | Low | High (crash at runtime) | Values derived from known HuggingFace config; code introspects KV structure from ONNX inputs as sanity check |
| EOS tokens incorrect | Low | Medium (model rambles or stops early) | Derived from tokenizer.json special tokens; standard ChatML tokens |
| Quantized q4f16 model has different KV cache dtype | Low | Medium | Code already handles Float16 via `extract_as_f32`; KV dtype detected dynamically from input element type |

## Goal Verification

### Truths
1. `cargo check` succeeds with no errors after all changes
2. `cargo build --bin xslm` produces a binary
3. `xslm.rs` contains Model enum, Input/Output/Handle/Listener types, and create() function matching slm.rs structure
4. `prompt.rs` contains a `build_xslm` function for xslm models
5. `lib.rs` exports the xslm module

### Artifacts
- `actor/src/xslm.rs` — library module
- `actor/src/bin/xslm.rs` — binary
- `actor/src/prompt.rs` — extended with `build_xslm`
- `actor/src/lib.rs` — module registration

### Key Links
- `xslm.rs` imports from `onnx` module (Session, Value, Executor types)
- `bin/xslm.rs` imports from `xslm` module and `prompt` module
- `prompt.rs` references `xslm::Model` enum
