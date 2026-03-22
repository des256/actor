Pocket TTS — Architecture & Execution Report

Overview

Pocket TTS is a streaming text-to-speech system built on 4 ONNX models and a tokenizer, running on a dedicated OS thread. It uses a flow-matching architecture
with a Mimi neural audio codec decoder. Audio is produced incrementally (frame by frame) and streamed to consumers via Tokio channels.

---

Model Files (all ONNX)

Model: Text Conditioner
Path: data/tts/pocket/text_conditioner.onnx
Quantization: Full precision
Purpose: Converts token IDs → text embeddings (1024-dim)
────────────────────────────────────────
Model: Flow LM Main
Path: data/tts/pocket/flow_lm_main_int8.onnx
Quantization: INT8
Purpose: Autoregressive backbone — produces conditioning vectors + EOS logit, maintains KV cache states
────────────────────────────────────────
Model: Flow LM Step
Path: data/tts/pocket/flow_lm_flow_int8.onnx
Quantization: INT8
Purpose: Flow-matching denoiser — refines noise → latent via learned vector field
────────────────────────────────────────
Model: Mimi Decoder
Path: data/tts/pocket/mimi_decoder_int8.onnx
Quantization: INT8
Purpose: Converts 32-dim latent → PCM audio frame, maintains streaming decoder states
────────────────────────────────────────
Model: Tokenizer
Path: data/tts/pocket/tokenizer.json
Quantization: N/A
Purpose: HuggingFace tokenizers JSON (sentencepiece-style, uses ▁ as word separator)

---

Constants

┌───────────────────────┬───────┬────────────────────────────────────────────────────┐
│ Name │ Value │ Description │
├───────────────────────┼───────┼────────────────────────────────────────────────────┤
│ MAX_TOKENS │ 1000 │ Maximum generation steps per utterance │
├───────────────────────┼───────┼────────────────────────────────────────────────────┤
│ LATENT_DIM │ 32 │ Dimensionality of the audio latent space │
├───────────────────────┼───────┼────────────────────────────────────────────────────┤
│ CONDITIONING_DIM │ 1024 │ Dimensionality of text/voice embeddings │
├───────────────────────┼───────┼────────────────────────────────────────────────────┤
│ DEFAULT_TEMPERATURE │ 0.7 │ Controls noise variance in flow-matching sampling │
├───────────────────────┼───────┼────────────────────────────────────────────────────┤
│ DEFAULT_LSD_STEPS │ 1 │ Number of Langevin-style denoising steps per token │
├───────────────────────┼───────┼────────────────────────────────────────────────────┤
│ DEFAULT_EOS_THRESHOLD │ -4.0 │ EOS logit threshold (above this = end of speech) │
└───────────────────────┴───────┴────────────────────────────────────────────────────┘

---

Voice File Format

Binary file (load_voice at pocket.rs:27):

[4 bytes] u32 LE: ndims (number of dimensions)
[ndims × 8 bytes] u64 LE per dimension
[remaining] f32 LE raw data

The voice data is loaded as flat Vec<f32>. During initialization, it's reshaped as [1, latent_frames, 1024] where latent_frames = voice.len() / 1024. This is
pre-computed speaker embedding data.

---

Initialization Sequence (create function, pocket.rs:404)

1. Load all 4 ONNX sessions — each with OptimizationLevel::EnableAll and 4 intra-op threads
2. Load tokenizer from JSON
3. Load voice latents from binary file
4. Discover stateful I/O names via get*names() — enumerates session inputs, excludes known fixed inputs (sequence/text_embeddings for flow_main, latent for
   decoder), and constructs output names as out*<input_name>. This is how KV cache state tensors are identified.
5. Initialize states via initialize_states() — creates zero-filled tensors matching each state input's shape and element type (f32, i64, or bool). States with
   shape [0] get empty tensors.
6. Condition on voice — runs flow_main once with:


    - sequence: empty tensor [1, 0, 32]
    - text_embeddings: voice data as [1, latent_frames, 1024]
    - Plus all state tensors
    - Output: updated state tensors → stored as reset_flow_states (the "warm" state after voice conditioning)

7. Pre-allocate reusable tensors for the generation loop:


    - sequence_tensor: [1, 1, 32] — holds one latent frame
    - empty_text_embeddings: [1, 0, 1024] — empty (no text conditioning during generation)
    - s_tensor, t_tensor: [1, 1] — flow step time boundaries
    - c_tensor: [1, 1024] — conditioning vector
    - x_tensor: [1, 32] — current latent being denoised
    - decoder_latent_tensor: [1, 1, 32] — decoder input

---

Per-Utterance Processing Pipeline

Runs on a dedicated std::thread. The main loop receives Stamped<TtsInput<T>> via std_mpsc::channel.

Step 1: Text Preparation (prepare, pocket.rs:179)

- Trim whitespace, replace newlines with spaces
- Count words to determine frames_after (trailing frames after EOS):
  - <= 4 words: 3 + 2 = 5 trailing frames
  - > 4 words: 1 + 2 = 3 trailing frames
- Short utterances (< 5 words): prepend 8 spaces (padding for the model)
- Replace all spaces with ▁ (U+2581, sentencepiece word-boundary token)
- Prepend a leading ▁

Example: "Hello world" → " Hello world" → "▁▁▁▁▁▁▁▁Hello▁world"

Step 2: Tokenization (tokenize, pocket.rs:200)

Uses HuggingFace tokenizers crate. Encodes the prepared string (without special tokens) → Vec<i64> token IDs.

Step 3: Text Conditioning (condition, pocket.rs:209)

1. Create tokens tensor: [1, seq_len] of i64
2. Run text_conditioner: token_ids → embeddings (shape: [1, seq_len, 1024])
3. Run flow_main with:


    - sequence: empty [1, 0, 32]
    - text_embeddings: the embeddings from step 2
    - All flow states (deep-cloned from reset_flow_states at utterance start)

4. Output: updated flow states (text is now "baked into" the KV cache)

Step 4: Autoregressive Generation Loop (up to 1000 steps)

Each iteration of the loop calls step() then decode_audio():

step() (pocket.rs:265):

1. Write current latent_state (32 floats, initialized to NaN) into the pre-allocated sequence_tensor
2. Run flow_main with:


    - sequence: [1, 1, 32] — the current latent
    - text_embeddings: empty [1, 0, 1024]
    - All flow states

3. Extract first two outputs: conditioning (1024-dim vector) and eos_logit (scalar). Remaining outputs become updated flow states.
4. Flow-matching denoising (LSD steps):


    - Sample initial noise: latent[i] ~ Normal(0, sqrt(temperature)) for each of 32 dimensions
    - For each of DEFAULT_LSD_STEPS (=1) steps:
        - Set s = i/num_steps, t = (i+1)/num_steps (time boundaries)
      - Copy conditioning into c_tensor, current latent into x_tensor
      - Run flow_step with (c, s, t, x) → flow_dir (32-dim vector field)
      - Update: latent[j] += flow_dir[j] / num_steps

5. Store final latent into latent_state and decoder_latent_tensor
6. Return eos_logit > DEFAULT_EOS_THRESHOLD as the EOS flag

decode_audio() (pocket.rs:363):

1. Run mimi_decoder with:


    - latent: [1, 1, 32] — the denoised latent
    - All decoder states

2. Extract audio_frame (f32). Remaining outputs become updated decoder states.
3. Convert f32 → i16: (f \* 32768.0).clamp(-32768.0, 32767.0) as i16

Step 5: Streaming Output

Each step produces one audio frame (sent as TtsOutput with Vec<i16> data). Frames are sent via tokio_mpsc::channel(32) with blocking_send.

Each frame gets an incrementing id (u64) and the last flag is set on the final frame.

Step 6: EOS Handling

When eos_logit > -4.0, an eos_countdown is set to frames_after. The loop continues generating that many more frames (allowing the model to produce trailing
audio), then sends the final frame with last: true and breaks.

---

Cancellation / Epoch System

- All messages are stamped with an epoch value (Stamped<T>)
- Epoch is a shared AtomicU64 counter
- On receipt, stale inputs (epoch doesn't match current) are skipped
- During generation, epoch.is_current(my_epoch) is checked each step — if the epoch has advanced (via epoch.advance() from outside), generation breaks
  immediately
- This enables instant cancellation of in-progress TTS when new input arrives

---

Public API

// Creation — returns handle (sender) and listener (receiver)
pub fn create<T>(onnx, executor, voice_path, epoch) -> (PocketHandle<T>, PocketListener<T>)

// Send text for synthesis
PocketHandle::send(TtsInput { payload: T, text: String })

// Receive streaming audio
PocketListener::recv() -> Option<Stamped<TtsOutput<T>>> // async
PocketListener::try_recv() -> Option<Stamped<TtsOutput<T>>> // non-blocking

The generic T is a user-defined payload that flows through untouched (for correlation).

---

Key Details for TensorRT Reimplementation

1. State management is central. Both flow*main and mimi_decoder are stateful (KV caches). State tensors are discovered dynamically by enumerating inputs,
   excluding known fixed ones, and expecting outputs named out*<input>. The TensorRT version needs equivalent stateful session management.
2. Voice conditioning happens once per voice load (produces reset_flow_states), then those states are deep-cloned per utterance. Text conditioning runs once
   per utterance. Only the generation loop runs per-frame.
3. Pre-allocated tensors (sequence_tensor, s_tensor, t_tensor, c_tensor, x_tensor, decoder_latent_tensor) are mutated in-place via as_slice_mut() each step to
   avoid allocation. TensorRT equivalent should use pinned/device memory for the same tensors.
4. The flow-matching step with DEFAULT_LSD_STEPS=1 is effectively a single Euler step from noise to latent. With more steps it would be iterative refinement.
   The s and t inputs are fractional timesteps.
5. Audio output is 24kHz 16-bit PCM (based on test assertions). Each decoder frame produces a variable number of samples.
6. The thread model is a single OS thread doing all 4 models synchronously in sequence. Input via std_mpsc, output via tokio_mpsc(32). The 32-slot buffer
   allows the generation thread to stay ahead of audio playback.
