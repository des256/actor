use {
    crate::*,
    std::{
        ffi::{CString, c_void},
        fs::File,
        io::Read,
        path::Path,
        ptr::null_mut,
        sync::{Arc, mpsc as std_mpsc},
    },
    tokenizers::Tokenizer,
    tokio::sync::mpsc as tokio_mpsc,
};

const TEXT_ENCODER_PATH: &str = "data/pocket/engine/text_encoder.engine";
const FLOW_LM_MAIN_PATH: &str = "data/pocket/engine/flow_lm_main.engine";
const FLOW_LM_FLOW_PATH: &str = "data/pocket/engine/flow_lm_flow.engine";
const MIMI_DECODER_PATH: &str = "data/pocket/engine/mimi_decoder.engine";
const TOKENIZER_PATH: &str = "data/pocket/engine/tokenizer.json";

const MAX_TOKENS: usize = 250;
const FLUSH_FRAMES: usize = 10;
const CHANNEL_CAPACITY: usize = 64;

// Model dimensions (from config/b6369a24.yaml)
const D_MODEL: usize = 1024;
const NUM_HEADS: usize = 16;
const NUM_LAYERS: usize = 6;
const HEAD_DIM: usize = D_MODEL / NUM_HEADS; // 64
const LDIM: usize = 32;
const TEMPERATURE: f32 = 0.7;
const EOS_THRESHOLD: f32 = -4.0;

// TRT buffer sizing — must match --maxShapes in build.sh
const MAX_TEXT_LEN: usize = 500;
const MAX_COND_LEN: usize = 500;
const MAX_PAST_LEN: usize = 800;
const SAMPLES_PER_FRAME: usize = 1920;
const KV_MAX_BYTES: usize =
    NUM_LAYERS * 2 * 1 * MAX_PAST_LEN * NUM_HEADS * HEAD_DIM * size_of::<f32>();
const LATENT_BYTES: usize = LDIM * size_of::<f32>();

// CUDA memcpy direction constants
const CUDA_MEMCPY_D2D: i32 = 3;

// Mimi decoder state buffer sizes (15 states, from export.py state table)
const MIMI_STATE_SIZES: [usize; 15] = [
    12288,   // state_0:  1*512*6*4
    6144,    // state_1:  1*256*6*4
    2048,    // state_2:  1*256*2*4
    2560,    // state_3:  1*128*5*4
    1024,    // state_4:  1*128*2*4
    1024,    // state_5:  1*64*4*4
    512,     // state_6:  1*64*2*4
    512,     // state_7:  1*64*2*4
    1024000, // state_8:  2*1*8*250*64*4
    8,       // state_9:  1*8 (i64)
    8,       // state_10: 1*8 (i64)
    1024000, // state_11: 2*1*8*250*64*4
    8,       // state_12: 1*8 (i64)
    8,       // state_13: 1*8 (i64)
    32768,   // state_14: 1*512*16*4
];

/// Pre-computed CString tensor names to avoid per-frame heap allocations.
struct TensorNames {
    // Text encoder
    te_token_ids: CString,
    te_text_emb: CString,
    // Flow LM Main
    main_sequence: CString,
    main_text_emb: CString,
    main_kv_cache: CString,
    main_cache_len: CString,
    main_conditioning: CString,
    main_eos_logit: CString,
    main_new_kv_cache: CString,
    main_new_cache_len: CString,
    // Flow LM Flow
    flow_c: CString,
    flow_s: CString,
    flow_t: CString,
    flow_x: CString,
    flow_latent: CString, // fused output: noise + flow_dir
    // Mimi decoder
    mimi_latent: CString,
    mimi_audio: CString,
    mimi_states: [CString; 15],
    mimi_out_states: [CString; 15],
}

impl TensorNames {
    fn new() -> Self {
        Self {
            te_token_ids: CString::new("token_ids").unwrap(),
            te_text_emb: CString::new("text_embeddings").unwrap(),
            main_sequence: CString::new("sequence").unwrap(),
            main_text_emb: CString::new("text_embeddings").unwrap(),
            main_kv_cache: CString::new("kv_cache").unwrap(),
            main_cache_len: CString::new("cache_len").unwrap(),
            main_conditioning: CString::new("conditioning").unwrap(),
            main_eos_logit: CString::new("eos_logit").unwrap(),
            main_new_kv_cache: CString::new("new_kv_cache").unwrap(),
            main_new_cache_len: CString::new("new_cache_len").unwrap(),
            flow_c: CString::new("c").unwrap(),
            flow_s: CString::new("s").unwrap(),
            flow_t: CString::new("t").unwrap(),
            flow_x: CString::new("x").unwrap(),
            flow_latent: CString::new("latent").unwrap(),
            mimi_latent: CString::new("latent").unwrap(),
            mimi_audio: CString::new("audio").unwrap(),
            mimi_states: std::array::from_fn(|i| CString::new(format!("state_{i}")).unwrap()),
            mimi_out_states: std::array::from_fn(|i| {
                CString::new(format!("out_state_{i}")).unwrap()
            }),
        }
    }
}

fn load_voice(voice_path: impl AsRef<Path>) -> Vec<f32> {
    let mut file = File::open(voice_path).unwrap();
    let mut buf4 = [0u8; 4];
    file.read_exact(&mut buf4).unwrap();
    let ndims = u32::from_le_bytes(buf4) as usize;
    let mut total_elements: usize = 1;
    for _ in 0..ndims {
        let mut buf8 = [0u8; 8];
        file.read_exact(&mut buf8).unwrap();
        let dim = u64::from_le_bytes(buf8) as usize;
        total_elements = total_elements.checked_mul(dim).unwrap();
    }
    let mut data = vec![0f32; total_elements];
    let slice = unsafe {
        std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, total_elements * 4)
    };
    file.read_exact(slice).unwrap();
    data
}

pub struct Input<T: Clone + Send + 'static> {
    pub payload: T,       // pass-along payload
    pub sentence: String, // sentence to generate from
    pub stamp: u64,       // epoch timestamp
}

pub struct Output<T: Clone + Send + 'static> {
    pub payload: T,      // pass-along payload
    pub audio: Vec<i16>, // generated audio chunk
    pub index: usize,    // chunk index for the current sentence
    pub last: bool,      // this is the last chunk of the current sentence
    pub stamp: u64,      // epoch timestamp
}

struct Direct {
    // TRT contexts
    te_ctx: Arc<tensorrt::Context>,
    main_ctx: Arc<tensorrt::Context>,
    flow_ctx: Arc<tensorrt::Context>,
    mimi_ctx: Arc<tensorrt::Context>,

    // CUDA streams
    stream: *mut c_void,        // main stream (step + conditioning)
    decode_stream: *mut c_void, // mimi decoder stream (overlaps with step)

    tokenizer: Tokenizer,
    names: TensorNames,

    // Text encoder GPU buffers
    te_token_ids: tensorrt::Buffer, // [1, text_len] i64
    te_text_emb: tensorrt::Buffer,  // [1, cond_len, D_MODEL] f32

    // Flow LM Main GPU buffers
    main_sequence: tensorrt::Buffer,      // [1, 1, LDIM] f32
    main_kv_a: tensorrt::Buffer,          // [6, 2, 1, past_len, 16, 64] f32
    main_kv_b: tensorrt::Buffer,          // [6, 2, 1, past_len, 16, 64] f32
    main_cache_len: tensorrt::Buffer,     // [1] i64
    main_conditioning: tensorrt::Buffer,  // [1, D_MODEL] f32
    main_eos_logit: tensorrt::Buffer,     // [1, 1] f32
    main_new_cache_len: tensorrt::Buffer, // [1] i64

    // Flow LM Flow GPU buffers
    flow_s: tensorrt::Buffer,     // [1, 1] f32 (constant 0.0)
    flow_t: tensorrt::Buffer,     // [1, 1] f32 (constant 1.0)
    flow_noise: tensorrt::Buffer, // [1, LDIM] f32

    // Double-buffered latent (fused flow output, also mimi decoder input)
    latent_a: tensorrt::Buffer, // [1, LDIM] f32
    latent_b: tensorrt::Buffer, // [1, LDIM] f32
    bos_buf: tensorrt::Buffer,  // [1, LDIM] f32 (pre-filled with NaN for BOS)

    // Mimi decoder GPU buffers
    mimi_audio: tensorrt::Buffer,         // [1, 1, SAMPLES_PER_FRAME] f32
    mimi_states_a: Vec<tensorrt::Buffer>, // 15 state buffers (A set)
    mimi_states_b: Vec<tensorrt::Buffer>, // 15 state buffers (B set)

    // Voice conditioning cache
    cached_kv: tensorrt::Buffer, // saved post-voice KV cache
    cached_cache_len: i64,
    has_cache: bool,

    // Pre-allocated host buffers
    noise_buf: [f32; LDIM],
    eos_buf: [f32; 1],
    audio_buf: [f32; SAMPLES_PER_FRAME],

    // Generation state
    cache_len: i64,
    kv_idx: bool,         // false → A is input / B is output
    mimi_state_idx: bool, // false → A is input / B is output
    latent_idx: bool,     // false → step writes A, true → step writes B
    has_prev_latent: bool,
    rng: Xorshift64,
}

impl Direct {
    pub fn new(
        te_engine: &Arc<tensorrt::Engine>,
        main_engine: &Arc<tensorrt::Engine>,
        flow_engine: &Arc<tensorrt::Engine>,
        mimi_engine: &Arc<tensorrt::Engine>,
        tokenizer: Tokenizer,
    ) -> Self {
        let te_ctx = te_engine.create_context();
        let main_ctx = main_engine.create_context();
        let flow_ctx = flow_engine.create_context();
        let mimi_ctx = mimi_engine.create_context();

        let mut stream: *mut c_void = null_mut();
        let rc = unsafe { tensorrt::ffi::cudaStreamCreate(&mut stream) };
        assert!(rc == 0, "cudaStreamCreate failed: {rc}");

        let mut decode_stream: *mut c_void = null_mut();
        let rc = unsafe { tensorrt::ffi::cudaStreamCreate(&mut decode_stream) };
        assert!(rc == 0, "cudaStreamCreate (decode) failed: {rc}");

        let names = TensorNames::new();
        let f32s = size_of::<f32>();
        let i64s = size_of::<i64>();

        // Text encoder buffers
        let te_token_ids = tensorrt::Buffer::new(MAX_TEXT_LEN * i64s);
        let te_text_emb = tensorrt::Buffer::new(MAX_COND_LEN * D_MODEL * f32s);

        // Flow LM Main buffers
        let main_sequence = tensorrt::Buffer::new(1 * LDIM * f32s);
        let main_kv_a = tensorrt::Buffer::new(KV_MAX_BYTES);
        let main_kv_b = tensorrt::Buffer::new(KV_MAX_BYTES);
        let main_cache_len = tensorrt::Buffer::new(i64s);
        let main_conditioning = tensorrt::Buffer::new(D_MODEL * f32s);
        let main_eos_logit = tensorrt::Buffer::new(f32s);
        let main_new_cache_len = tensorrt::Buffer::new(i64s);

        // Flow LM Flow buffers
        let flow_s = tensorrt::Buffer::new(f32s);
        let flow_t = tensorrt::Buffer::new(f32s);
        let flow_noise = tensorrt::Buffer::new(LDIM * f32s);

        // Pre-upload constants
        flow_s.upload(&[0.0f32]);
        flow_t.upload(&[1.0f32]);

        // Double-buffered latent (fused flow output + mimi input)
        let latent_a = tensorrt::Buffer::new(LATENT_BYTES);
        let latent_b = tensorrt::Buffer::new(LATENT_BYTES);

        // BOS buffer (NaN marker for first step)
        let bos_buf = tensorrt::Buffer::new(LATENT_BYTES);
        bos_buf.upload(&[f32::NAN; LDIM]);

        // Mimi decoder buffers
        let mimi_audio = tensorrt::Buffer::new(SAMPLES_PER_FRAME * f32s);

        // Mimi decoder state buffers (15 states × 2 sets for double-buffering)
        let mimi_states_a: Vec<tensorrt::Buffer> = MIMI_STATE_SIZES
            .iter()
            .map(|&size| tensorrt::Buffer::new(size))
            .collect();
        let mimi_states_b: Vec<tensorrt::Buffer> = MIMI_STATE_SIZES
            .iter()
            .map(|&size| tensorrt::Buffer::new(size))
            .collect();

        // Voice conditioning cache buffer
        let cached_kv = tensorrt::Buffer::new(KV_MAX_BYTES);

        // Set fixed mimi decoder input shape (never changes — always single frame)
        mimi_ctx.set_input_shape("latent", &[1, LDIM as i64, 1]);

        Self {
            te_ctx,
            main_ctx,
            flow_ctx,
            mimi_ctx,
            stream,
            decode_stream,
            tokenizer,
            names,
            te_token_ids,
            te_text_emb,
            main_sequence,
            main_kv_a,
            main_kv_b,
            main_cache_len,
            main_conditioning,
            main_eos_logit,
            main_new_cache_len,
            flow_s,
            flow_t,
            flow_noise,
            latent_a,
            latent_b,
            bos_buf,
            mimi_audio,
            mimi_states_a,
            mimi_states_b,
            cached_kv,
            cached_cache_len: 0,
            has_cache: false,
            noise_buf: [0.0; LDIM],
            eos_buf: [0.0; 1],
            audio_buf: [0.0; SAMPLES_PER_FRAME],
            cache_len: 0,
            kv_idx: false,
            mimi_state_idx: false,
            latent_idx: false,
            has_prev_latent: false,
            rng: Xorshift64::new(42),
        }
    }

    /// Run a conditioning pass through flow_lm_main (empty sequence input).
    ///
    /// The conditioning data (voice or text embeddings) must already be
    /// uploaded to `te_text_emb` on the GPU with shape [1, cond_len, D_MODEL].
    fn conditioning_pass(&mut self, cond_len: usize) {
        self.main_ctx
            .set_input_shape_cstr(&self.names.main_sequence, &[1, 0, LDIM as i64]);
        self.main_ctx.set_input_shape_cstr(
            &self.names.main_text_emb,
            &[1, cond_len as i64, D_MODEL as i64],
        );
        self.main_ctx.set_input_shape_cstr(
            &self.names.main_kv_cache,
            &[
                NUM_LAYERS as i64,
                2,
                1,
                self.cache_len,
                NUM_HEADS as i64,
                HEAD_DIM as i64,
            ],
        );

        self.main_cache_len.upload(&[self.cache_len]);

        let (kv_in, kv_out) = if !self.kv_idx {
            (&self.main_kv_a, &self.main_kv_b)
        } else {
            (&self.main_kv_b, &self.main_kv_a)
        };

        self.main_ctx
            .set_tensor_address_cstr(&self.names.main_sequence, self.main_sequence.ptr);
        self.main_ctx
            .set_tensor_address_cstr(&self.names.main_text_emb, self.te_text_emb.ptr);
        self.main_ctx
            .set_tensor_address_cstr(&self.names.main_kv_cache, kv_in.ptr);
        self.main_ctx
            .set_tensor_address_cstr(&self.names.main_cache_len, self.main_cache_len.ptr);
        self.main_ctx
            .set_tensor_address_cstr(&self.names.main_conditioning, self.main_conditioning.ptr);
        self.main_ctx
            .set_tensor_address_cstr(&self.names.main_eos_logit, self.main_eos_logit.ptr);
        self.main_ctx
            .set_tensor_address_cstr(&self.names.main_new_kv_cache, kv_out.ptr);
        self.main_ctx
            .set_tensor_address_cstr(&self.names.main_new_cache_len, self.main_new_cache_len.ptr);

        self.main_ctx.enqueue(self.stream);
        unsafe { tensorrt::ffi::cudaStreamSynchronize(self.stream) };

        self.cache_len += cond_len as i64;
        self.kv_idx = !self.kv_idx;
    }

    fn reset(&mut self, voice: &[f32]) {
        if self.has_cache {
            // Restore from cached post-voice state (skip voice conditioning pass)
            // Copy cached KV to A, set kv_idx=false so A is the next input
            self.main_kv_a
                .copy_from_device(&self.cached_kv, KV_MAX_BYTES);
            self.kv_idx = false;
            self.cache_len = self.cached_cache_len;
            // Zero mimi states A (decoder reads from A on first frame)
            for (i, buf) in self.mimi_states_a.iter().enumerate() {
                unsafe { tensorrt::ffi::cudaMemset(buf.ptr, 0, MIMI_STATE_SIZES[i]) };
            }
            self.mimi_state_idx = false;
            self.latent_idx = false;
            self.has_prev_latent = false;
            self.rng = Xorshift64::new(42);
            return;
        }

        // First call: full reset + voice conditioning
        unsafe { tensorrt::ffi::cudaMemset(self.main_kv_a.ptr, 0, KV_MAX_BYTES) };
        for (i, buf) in self.mimi_states_a.iter().enumerate() {
            unsafe { tensorrt::ffi::cudaMemset(buf.ptr, 0, MIMI_STATE_SIZES[i]) };
        }
        self.cache_len = 0;
        self.kv_idx = false;
        self.mimi_state_idx = false;
        self.latent_idx = false;
        self.has_prev_latent = false;
        self.rng = Xorshift64::new(42);

        // Voice conditioning pass
        if !voice.is_empty() {
            let t_frames = voice.len() / D_MODEL;
            self.te_text_emb.upload(voice);
            self.conditioning_pass(t_frames);

            // Cache post-voice state for subsequent resets.
            // After conditioning_pass: kv_idx=true, active KV is in B (B is input).
            // Save B to cached_kv. On restore, we copy to A and set kv_idx=false.
            self.cached_kv
                .copy_from_device(&self.main_kv_b, KV_MAX_BYTES);
            self.cached_cache_len = self.cache_len;
            self.has_cache = true;
        }
    }

    fn prepare(&mut self, sentence: &str) -> (Vec<i64>, usize) {
        let text = prepare_text(sentence);
        let encoding = match self.tokenizer.encode(text, false) {
            Ok(enc) => enc,
            Err(e) => {
                eprintln!("pocket: tokenization failed for {:?}: {}", sentence, e);
                return (vec![], 0);
            }
        };
        let tokens: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

        let word_count = sentence.split_whitespace().count();
        let eos_countdown_seed = if word_count <= 4 { 5 } else { 3 };

        (tokens, eos_countdown_seed)
    }

    fn condition(&mut self, tokens: &[i64]) {
        let num_tokens = tokens.len();

        // Run text_encoder: token_ids [1, T] → text_embeddings [1, T, D_MODEL]
        self.te_token_ids.upload(tokens);

        self.te_ctx
            .set_input_shape_cstr(&self.names.te_token_ids, &[1, num_tokens as i64]);
        self.te_ctx
            .set_tensor_address_cstr(&self.names.te_token_ids, self.te_token_ids.ptr);
        self.te_ctx
            .set_tensor_address_cstr(&self.names.te_text_emb, self.te_text_emb.ptr);

        self.te_ctx.enqueue(self.stream);
        unsafe { tensorrt::ffi::cudaStreamSynchronize(self.stream) };

        // Text conditioning pass (text embeddings now in te_text_emb)
        self.conditioning_pass(num_tokens);
    }

    /// Launch flow_lm_main + flow_lm_flow on the main stream (async).
    ///
    /// Handles BOS (first step) and D2D latent copy (subsequent steps)
    /// for the sequence input. Generates noise while flow_lm_main runs
    /// on GPU, then enqueues flow_lm_flow. Updates cache_len, kv_idx,
    /// latent_idx deterministically (no sync needed).
    fn step_launch(&mut self) {
        // Sequence input: D2D copy previous latent or BOS to main_sequence
        let seq_src = if self.has_prev_latent {
            // Previous latent is at the buffer indicated by latent_idx
            // (latent_idx points to the last-written buffer after prior step flipped it)
            if self.latent_idx {
                self.latent_a.ptr
            } else {
                self.latent_b.ptr
            }
        } else {
            self.bos_buf.ptr
        };
        unsafe {
            tensorrt::ffi::cudaMemcpyAsync(
                self.main_sequence.ptr,
                seq_src as *const c_void,
                LATENT_BYTES,
                CUDA_MEMCPY_D2D,
                self.stream,
            );
        }

        // Set flow_lm_main shapes
        self.main_ctx
            .set_input_shape_cstr(&self.names.main_sequence, &[1, 1, LDIM as i64]);
        self.main_ctx
            .set_input_shape_cstr(&self.names.main_text_emb, &[1, 0, D_MODEL as i64]);
        self.main_ctx.set_input_shape_cstr(
            &self.names.main_kv_cache,
            &[
                NUM_LAYERS as i64,
                2,
                1,
                self.cache_len,
                NUM_HEADS as i64,
                HEAD_DIM as i64,
            ],
        );

        // Upload cache_len (async on main_stream, ordered before enqueue)
        self.main_cache_len
            .upload_async(&[self.cache_len], self.stream);

        // Set tensor addresses
        let (kv_in, kv_out) = if !self.kv_idx {
            (self.main_kv_a.ptr, self.main_kv_b.ptr)
        } else {
            (self.main_kv_b.ptr, self.main_kv_a.ptr)
        };

        self.main_ctx
            .set_tensor_address_cstr(&self.names.main_sequence, self.main_sequence.ptr);
        self.main_ctx
            .set_tensor_address_cstr(&self.names.main_text_emb, self.te_text_emb.ptr);
        self.main_ctx
            .set_tensor_address_cstr(&self.names.main_kv_cache, kv_in);
        self.main_ctx
            .set_tensor_address_cstr(&self.names.main_cache_len, self.main_cache_len.ptr);
        self.main_ctx
            .set_tensor_address_cstr(&self.names.main_conditioning, self.main_conditioning.ptr);
        self.main_ctx
            .set_tensor_address_cstr(&self.names.main_eos_logit, self.main_eos_logit.ptr);
        self.main_ctx
            .set_tensor_address_cstr(&self.names.main_new_kv_cache, kv_out);
        self.main_ctx
            .set_tensor_address_cstr(&self.names.main_new_cache_len, self.main_new_cache_len.ptr);

        // Enqueue flow_lm_main
        self.main_ctx.enqueue(self.stream);

        // Generate noise on CPU (overlaps with flow_lm_main GPU execution)
        let temp_scale = TEMPERATURE.sqrt();
        for i in (0..LDIM).step_by(2) {
            let (n1, n2) = self.rng.normal_pair();
            self.noise_buf[i] = n1 * temp_scale;
            self.noise_buf[i + 1] = n2 * temp_scale;
        }

        // Upload noise (async on main_stream, ordered after flow_lm_main)
        self.flow_noise
            .upload_async(&self.noise_buf, self.stream);

        // Set flow_lm_flow shapes and addresses
        self.flow_ctx
            .set_input_shape_cstr(&self.names.flow_c, &[1, D_MODEL as i64]);
        self.flow_ctx
            .set_input_shape_cstr(&self.names.flow_s, &[1, 1]);
        self.flow_ctx
            .set_input_shape_cstr(&self.names.flow_t, &[1, 1]);
        self.flow_ctx
            .set_input_shape_cstr(&self.names.flow_x, &[1, LDIM as i64]);

        self.flow_ctx
            .set_tensor_address_cstr(&self.names.flow_c, self.main_conditioning.ptr);
        self.flow_ctx
            .set_tensor_address_cstr(&self.names.flow_s, self.flow_s.ptr);
        self.flow_ctx
            .set_tensor_address_cstr(&self.names.flow_t, self.flow_t.ptr);
        self.flow_ctx
            .set_tensor_address_cstr(&self.names.flow_x, self.flow_noise.ptr);

        // Fused output: latent = noise + flow_dir (computed on GPU)
        let latent_out = if !self.latent_idx {
            self.latent_a.ptr
        } else {
            self.latent_b.ptr
        };
        self.flow_ctx
            .set_tensor_address_cstr(&self.names.flow_latent, latent_out);

        // Enqueue flow_lm_flow (same stream, ordered after flow_lm_main + noise upload)
        self.flow_ctx.enqueue(self.stream);

        // Update state (deterministic — no GPU sync needed)
        self.cache_len += 1;
        self.kv_idx = !self.kv_idx;
        self.latent_idx = !self.latent_idx;
        self.has_prev_latent = true;
    }

    /// Launch mimi decoder on the decode stream (async).
    ///
    /// Reads from the latent buffer that step_launch just wrote to
    /// (determined by latent_idx after flip).
    fn decode_launch(&mut self) {
        // The just-written latent: after step_launch flipped latent_idx,
        // the written buffer is at the PRE-flip index = current opposite.
        // if latent_idx=true → A was written (was false, wrote A, flipped to true)
        // if latent_idx=false → B was written (was true, wrote B, flipped to false)
        let latent_ptr = if self.latent_idx {
            self.latent_a.ptr
        } else {
            self.latent_b.ptr
        };

        self.mimi_ctx
            .set_tensor_address_cstr(&self.names.mimi_latent, latent_ptr);
        self.mimi_ctx
            .set_tensor_address_cstr(&self.names.mimi_audio, self.mimi_audio.ptr);

        // Double-buffer state swap
        let (state_in, state_out) = if self.mimi_state_idx {
            (&self.mimi_states_b, &self.mimi_states_a)
        } else {
            (&self.mimi_states_a, &self.mimi_states_b)
        };

        for i in 0..15 {
            self.mimi_ctx
                .set_tensor_address_cstr(&self.names.mimi_states[i], state_in[i].ptr);
            self.mimi_ctx
                .set_tensor_address_cstr(&self.names.mimi_out_states[i], state_out[i].ptr);
        }

        self.mimi_ctx.enqueue(self.decode_stream);
        self.mimi_state_idx = !self.mimi_state_idx;
    }
}

impl Drop for Direct {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe { tensorrt::ffi::cudaStreamDestroy(self.stream) };
        }
        if !self.decode_stream.is_null() {
            unsafe { tensorrt::ffi::cudaStreamDestroy(self.decode_stream) };
        }
    }
}

/// Simple text normalization matching Python's prepare_text_prompt().
fn prepare_text(text: &str) -> String {
    let mut t = text
        .trim()
        .replace('\n', " ")
        .replace('\r', " ")
        .replace("  ", " ");

    if t.is_empty() {
        return t;
    }

    // Uppercase first letter
    let mut chars = t.chars();
    if let Some(first) = chars.next() {
        t = first.to_uppercase().to_string() + chars.as_str();
    }

    // Ensure ends with punctuation
    if t.ends_with(|c: char| c.is_alphanumeric()) {
        t.push('.');
    }

    // Pad short texts
    if t.split_whitespace().count() < 5 {
        t = format!("        {}", t);
    }

    t
}

/// Xorshift64 PRNG with Box-Muller normal generation.
struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn next_f32_open(&mut self) -> f32 {
        // (0, 1) open interval
        ((self.next_u64() >> 40) as f32 + 0.5) / 16777216.0
    }

    fn normal_pair(&mut self) -> (f32, f32) {
        let u1 = self.next_f32_open();
        let u2 = self.next_f32_open();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        (r * theta.cos(), r * theta.sin())
    }
}

pub struct Handle<T: Clone + Send + 'static> {
    tx: std_mpsc::Sender<Input<T>>,
}

pub struct Listener<T: Clone + Send + 'static> {
    rx: tokio_mpsc::Receiver<Output<T>>,
}

pub fn create<T: Clone + Send + 'static>(
    trt: &Arc<tensorrt::Tensorrt>,
    voice_path: impl AsRef<Path>,
    epoch: &Arc<Epoch>,
) -> (Handle<T>, Listener<T>) {
    let (input_tx, input_rx) = std_mpsc::channel::<pocket::Input<T>>();
    let (output_tx, output_rx) = tokio_mpsc::channel::<pocket::Output<T>>(CHANNEL_CAPACITY);

    let te_engine = trt.load_engine(TEXT_ENCODER_PATH);
    let main_engine = trt.load_engine(FLOW_LM_MAIN_PATH);
    let flow_engine = trt.load_engine(FLOW_LM_FLOW_PATH);
    let mimi_engine = trt.load_engine(MIMI_DECODER_PATH);
    let tokenizer = Tokenizer::from_file(TOKENIZER_PATH).unwrap();
    let voice = load_voice(voice_path);

    std::thread::spawn({
        let epoch = Arc::clone(&epoch);
        move || {
            let mut d =
                Direct::new(&te_engine, &main_engine, &flow_engine, &mimi_engine, tokenizer);
            while let Ok(input) = input_rx.recv() {
                if !epoch.is_current(input.stamp) {
                    continue;
                }

                // Reset to post-voice state (uses cache after first call)
                d.reset(&voice);

                // Tokenize and condition
                let (tokens, eos_countdown_seed) = d.prepare(&input.sentence);
                if tokens.is_empty() {
                    eprintln!("pocket: skipping empty token sequence for {:?}", &input.sentence);
                    let _ = output_tx.blocking_send(Output {
                        payload: input.payload.clone(),
                        audio: vec![],
                        index: 0,
                        last: true,
                        stamp: input.stamp,
                    });
                    continue;
                }
                d.condition(&tokens);

                // === Two-stream pipeline ===
                //
                // Prologue: step(0) synchronous
                d.step_launch();
                unsafe { tensorrt::ffi::cudaStreamSynchronize(d.stream) };
                d.main_eos_logit.download(&mut d.eos_buf);
                let mut eos_step: Option<usize> =
                    if d.eos_buf[0] > EOS_THRESHOLD { Some(0) } else { None };

                let mut chunk_index = 0;
                for frame in 0..MAX_TOKENS {
                    if !epoch.is_current(input.stamp) {
                        break;
                    }

                    let is_last = frame + 1 >= MAX_TOKENS
                        || eos_step.is_some_and(|es| frame >= es + eos_countdown_seed);

                    // Launch decode(frame) on decode_stream.
                    // step(frame) is already complete (sync'd above or at end of prev iter).
                    d.decode_launch();

                    // Overlap: launch step(frame+1) on main_stream while
                    // decode(frame) runs on decode_stream.
                    if !is_last {
                        d.step_launch();
                    }

                    // Sync decode_stream and download audio
                    unsafe { tensorrt::ffi::cudaStreamSynchronize(d.decode_stream) };
                    d.mimi_audio.download(&mut d.audio_buf);

                    let audio: Vec<i16> = d
                        .audio_buf
                        .iter()
                        .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
                        .collect();

                    if let Err(error) = output_tx.blocking_send(Output {
                        payload: input.payload.clone(),
                        audio,
                        index: chunk_index,
                        last: is_last,
                        stamp: input.stamp,
                    }) {
                        panic!("Tts: failed to send output: {}", error);
                    }
                    chunk_index += 1;

                    if is_last {
                        // Ensure any in-flight step completes before next sentence
                        unsafe { tensorrt::ffi::cudaStreamSynchronize(d.stream) };
                        break;
                    }

                    // Sync main_stream and download EOS from step(frame+1)
                    unsafe { tensorrt::ffi::cudaStreamSynchronize(d.stream) };
                    d.main_eos_logit.download(&mut d.eos_buf);
                    if d.eos_buf[0] > EOS_THRESHOLD && eos_step.is_none() {
                        eos_step = Some(frame + 1);
                    }
                }


            }
        }
    });

    (Handle { tx: input_tx }, Listener { rx: output_rx })
}

impl<T: Clone + Send + 'static> Handle<T> {
    pub fn send(&self, input: Input<T>) {
        self.tx.send(input).unwrap();
    }
}

impl<T: Clone + Send + 'static> Listener<T> {
    pub async fn recv(&mut self) -> Output<T> {
        self.rx.recv().await.unwrap()
    }

    pub fn try_recv(&mut self) -> Option<Output<T>> {
        match self.rx.try_recv() {
            Ok(output) => Some(output),
            Err(tokio_mpsc::error::TryRecvError::Empty) => None,
            Err(tokio_mpsc::error::TryRecvError::Disconnected) => {
                panic!("Tts: output channel disconnected")
            }
        }
    }
}
