use {
    super::*,
    crate::tensorrt::{self, Buffer},
    std::{
        ffi::c_void,
        fs::File,
        io::{Read, Seek, SeekFrom},
        sync::Arc,
    },
};

// ── Architecture constants (from config/b6369a24.yaml) ──────────────────

const D_MODEL: usize = 1024;
const NUM_HEADS: usize = 16;
const NUM_LAYERS: usize = 6;
const HEAD_DIM: usize = D_MODEL / NUM_HEADS; // 64

// ── Buffer sizing — must match --maxShapes in build_engine.sh ───────────

/// Maximum input_emb seq_len for backbone (voice or text conditioning).
const MAX_INPUT_LEN: usize = 200;

/// Maximum KV cache past_len (backbone).
const MAX_PAST_LEN: usize = 500;

/// Maximum mimi_decoder input frames.
const MAX_MIMI_FRAMES: usize = 100;

/// Maximum audio output samples from mimi_decoder.
/// Each frame at 12.5 Hz ≈ 1920 samples at 24 kHz.
const MAX_AUDIO_SAMPLES: usize = MAX_MIMI_FRAMES * 2000;

/// KV cache bytes per tensor (k or v) at maximum past_len.
/// Shape: [NUM_LAYERS, 1, NUM_HEADS, past_len, HEAD_DIM] f32
const KV_BYTES: usize = NUM_LAYERS * 1 * NUM_HEADS * MAX_PAST_LEN * HEAD_DIM * size_of::<f32>();

// ── CUDA stream helpers ─────────────────────────────────────────────────

unsafe extern "C" {
    fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
}

// ── Metadata from safetensors ───────────────────────────────────────────

struct Metadata {
    embed_weight: Vec<f32>,        // [4001, 1024]
    input_linear_weight: Vec<f32>, // [1024, 32]
    emb_mean: Vec<f32>,            // [32]
    emb_std: Vec<f32>,             // [32]
    bos_emb: Vec<f32>,             // [32]
}

impl Metadata {
    fn load(path: &str) -> Self {
        let mut file = File::open(path).unwrap_or_else(|e| panic!("metadata open {path}: {e}"));
        let mut buf8 = [0u8; 8];
        file.read_exact(&mut buf8).unwrap();
        let header_size = u64::from_le_bytes(buf8) as usize;
        let mut header_bytes = vec![0u8; header_size];
        file.read_exact(&mut header_bytes).unwrap();
        let header_str = std::str::from_utf8(&header_bytes).unwrap();
        let data_offset = 8 + header_size;

        // Simple JSON parsing for safetensors header — extract offset pairs
        fn parse_offsets(json: &str, key: &str) -> (usize, usize) {
            let needle = format!("\"{}\"", key);
            let pos = json.find(&needle).unwrap_or_else(|| panic!("key {key} not found"));
            let rest = &json[pos..];
            let do_start = rest.find("\"data_offsets\"").unwrap();
            let bracket = rest[do_start..].find('[').unwrap();
            let close = rest[do_start + bracket..].find(']').unwrap();
            let nums = &rest[do_start + bracket + 1..do_start + bracket + close];
            let parts: Vec<usize> = nums.split(',').map(|s| s.trim().parse().unwrap()).collect();
            (parts[0], parts[1])
        }

        fn read_f32(file: &mut File, data_offset: usize, start: usize, end: usize) -> Vec<f32> {
            let byte_len = end - start;
            let count = byte_len / 4;
            file.seek(SeekFrom::Start((data_offset + start) as u64)).unwrap();
            let mut bytes = vec![0u8; byte_len];
            file.read_exact(&mut bytes).unwrap();
            let mut result = vec![0f32; count];
            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                result[i] = f32::from_le_bytes(chunk.try_into().unwrap());
            }
            result
        }

        let (s, e) = parse_offsets(header_str, "bos_emb");
        let bos_emb = read_f32(&mut file, data_offset, s, e);

        let (s, e) = parse_offsets(header_str, "conditioner.embed.weight");
        let embed_weight = read_f32(&mut file, data_offset, s, e);

        let (s, e) = parse_offsets(header_str, "emb_mean");
        let emb_mean = read_f32(&mut file, data_offset, s, e);

        let (s, e) = parse_offsets(header_str, "emb_std");
        let emb_std = read_f32(&mut file, data_offset, s, e);

        let (s, e) = parse_offsets(header_str, "input_linear.weight");
        let input_linear_weight = read_f32(&mut file, data_offset, s, e);

        Self { embed_weight, input_linear_weight, emb_mean, emb_std, bos_emb }
    }

    /// Look up token embeddings: token_ids → [T, D_MODEL]
    fn embed_tokens(&self, token_ids: &[i64]) -> Vec<f32> {
        let mut result = Vec::with_capacity(token_ids.len() * D_MODEL);
        for &id in token_ids {
            let start = (id as usize) * D_MODEL;
            result.extend_from_slice(&self.embed_weight[start..start + D_MODEL]);
        }
        result
    }

    /// Project latent [LDIM] → [D_MODEL] via input_linear (matmul)
    /// input_linear.weight is [D_MODEL, LDIM], so output[i] = Σ_j weight[i*LDIM+j] * latent[j]
    fn project_latent(&self, latent: &[f32]) -> Vec<f32> {
        let mut result = vec![0f32; D_MODEL];
        for i in 0..D_MODEL {
            let mut sum = 0f32;
            let row = i * LATENT_DIM;
            for j in 0..LATENT_DIM {
                sum += self.input_linear_weight[row + j] * latent[j];
            }
            result[i] = sum;
        }
        result
    }

    /// De-normalize latent from flow space to mimi space: x * std + mean
    fn denormalize_latent(&self, latent: &[f32]) -> Vec<f32> {
        latent
            .iter()
            .enumerate()
            .map(|(i, &v)| v * self.emb_std[i] + self.emb_mean[i])
            .collect()
    }
}

// ── TRT Pocket ──────────────────────────────────────────────────────────

pub struct TrtPocket {
    // CUDA stream
    stream: *mut c_void,

    // Backbone GPU buffers
    bb_input_emb: Buffer, // [1, seq_len, D_MODEL]
    bb_k_a: Buffer,       // [6, 1, 16, past_len, 64] — double buffer A
    bb_k_b: Buffer,       // [6, 1, 16, past_len, 64] — double buffer B
    bb_v_a: Buffer,       // same shape as k
    bb_v_b: Buffer,
    bb_hidden: Buffer,    // [1, D_MODEL] — shared with flow_net conditioning input
    bb_eos: Buffer,       // [1, 1]

    // Flow net GPU buffers
    flow_s: Buffer,     // [1, 1] (constant 0.0)
    flow_t: Buffer,     // [1, 1] (constant 1.0)
    flow_x: Buffer,     // [1, LDIM]
    flow_v: Buffer,     // [1, LDIM]

    // Mimi decoder GPU buffers
    mimi_latent: Buffer, // [1, LDIM, num_frames]
    mimi_audio: Buffer,  // [1, 1, audio_samples]

    // TRT contexts (hold Arc<Engine> which holds Arc<Tensorrt>)
    bb_ctx: Arc<tensorrt::Context>,
    flow_ctx: Arc<tensorrt::Context>,
    mimi_ctx: Arc<tensorrt::Context>,

    // CPU-side metadata
    meta: Metadata,

    // Generation state
    past_len: i64,
    kv_idx: bool, // false → A is input, B is output
    prev_latent: Option<Vec<f32>>,
    latents: Vec<Vec<f32>>,
    rng: Xorshift64,
}

unsafe impl Send for TrtPocket {}

impl TrtPocket {
    pub fn new(trt: &Arc<tensorrt::Tensorrt>) -> Self {
        let bb_engine = trt.load_engine(BACKBONE_PATH);
        let flow_engine = trt.load_engine(FLOW_NET_PATH);
        let mimi_engine = trt.load_engine(DECODER_PATH);

        let bb_ctx = bb_engine.create_context();
        let flow_ctx = flow_engine.create_context();
        let mimi_ctx = mimi_engine.create_context();

        let mut stream: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { cudaStreamCreate(&mut stream) };
        assert!(rc == 0, "cudaStreamCreate failed: {rc}");

        let f = size_of::<f32>();

        // Backbone buffers
        let bb_input_emb = Buffer::new(MAX_INPUT_LEN * D_MODEL * f);
        let bb_k_a = Buffer::new(KV_BYTES);
        let bb_k_b = Buffer::new(KV_BYTES);
        let bb_v_a = Buffer::new(KV_BYTES);
        let bb_v_b = Buffer::new(KV_BYTES);
        let bb_hidden = Buffer::new(D_MODEL * f);
        let bb_eos = Buffer::new(f);

        // Flow net buffers
        let flow_s = Buffer::new(f);
        let flow_t = Buffer::new(f);
        let flow_x = Buffer::new(LATENT_DIM * f);
        let flow_v = Buffer::new(LATENT_DIM * f);
        flow_s.upload(&[0.0f32]);
        flow_t.upload(&[1.0f32]);

        // Mimi decoder buffers
        let mimi_latent = Buffer::new(LATENT_DIM * MAX_MIMI_FRAMES * f);
        let mimi_audio = Buffer::new(MAX_AUDIO_SAMPLES * f);

        let meta = Metadata::load(METADATA_PATH);

        Self {
            stream,
            bb_input_emb,
            bb_k_a,
            bb_k_b,
            bb_v_a,
            bb_v_b,
            bb_hidden,
            bb_eos,
            flow_s,
            flow_t,
            flow_x,
            flow_v,
            mimi_latent,
            mimi_audio,
            bb_ctx,
            flow_ctx,
            mimi_ctx,
            meta,
            past_len: 0,
            kv_idx: false,
            prev_latent: None,
            latents: Vec::new(),
            rng: Xorshift64::new(42),
        }
    }

    /// Run backbone with input_emb, updating KV cache.
    fn run_backbone(&mut self, seq_len: usize) {
        self.bb_ctx
            .set_input_shape("input_emb", &[1, seq_len as i64, D_MODEL as i64]);
        self.bb_ctx.set_input_shape(
            "k_self",
            &[NUM_LAYERS as i64, 1, NUM_HEADS as i64, self.past_len, HEAD_DIM as i64],
        );
        self.bb_ctx.set_input_shape(
            "v_self",
            &[NUM_LAYERS as i64, 1, NUM_HEADS as i64, self.past_len, HEAD_DIM as i64],
        );

        let (k_in, k_out) = if !self.kv_idx {
            (&self.bb_k_a, &self.bb_k_b)
        } else {
            (&self.bb_k_b, &self.bb_k_a)
        };
        let (v_in, v_out) = if !self.kv_idx {
            (&self.bb_v_a, &self.bb_v_b)
        } else {
            (&self.bb_v_b, &self.bb_v_a)
        };

        self.bb_ctx.set_tensor_address("input_emb", self.bb_input_emb.ptr);
        self.bb_ctx.set_tensor_address("k_self", k_in.ptr);
        self.bb_ctx.set_tensor_address("v_self", v_in.ptr);
        self.bb_ctx.set_tensor_address("hidden", self.bb_hidden.ptr);
        self.bb_ctx.set_tensor_address("is_eos", self.bb_eos.ptr);
        self.bb_ctx.set_tensor_address("out_k_self", k_out.ptr);
        self.bb_ctx.set_tensor_address("out_v_self", v_out.ptr);

        self.bb_ctx.enqueue(self.stream);
        unsafe { cudaStreamSynchronize(self.stream) };

        self.past_len += seq_len as i64;
        self.kv_idx = !self.kv_idx;
    }

    /// Feed voice conditioning through backbone to populate KV cache.
    pub fn init_voice(&mut self, voice: &[f32]) {
        let num_frames = voice.len() / CONDITIONING_DIM;
        if num_frames == 0 {
            return;
        }
        // Voice embeddings are already in D_MODEL space [T, 1024]
        self.bb_input_emb.upload(voice);
        self.run_backbone(num_frames);
    }

    /// Feed text token embeddings through backbone to populate KV cache.
    pub fn condition(&mut self, tokens: &[i64]) {
        let embeddings = self.meta.embed_tokens(tokens);
        self.bb_input_emb.upload(&embeddings);
        self.run_backbone(tokens.len());
    }

    /// Reset generation state (restore to post-voice KV cache).
    pub fn reset(&mut self, voice: &[f32]) {
        self.past_len = 0;
        self.kv_idx = false;
        self.prev_latent = None;
        self.latents.clear();
        self.rng = Xorshift64::new(42);

        // Re-initialize voice conditioning
        self.init_voice(voice);
    }

    /// Run one autoregressive step. Returns (latent, is_eos).
    pub fn step(&mut self) -> (Vec<f32>, bool) {
        // Project previous latent (or BOS) to D_MODEL
        let input_emb = match &self.prev_latent {
            None => self.meta.project_latent(&self.meta.bos_emb),
            Some(latent) => self.meta.project_latent(latent),
        };
        self.bb_input_emb.upload(&input_emb);

        // Run backbone
        self.run_backbone(1);

        // Download EOS logit
        let mut eos_logit = [0f32; 1];
        self.bb_eos.download(&mut eos_logit);
        let is_eos = eos_logit[0] > DEFAULT_EOS_THRESHOLD;

        // Generate noise
        let temp_scale = DEFAULT_TEMPERATURE.sqrt();
        let noise: Vec<f32> = (0..LATENT_DIM / 2)
            .flat_map(|_| {
                let (n1, n2) = self.rng.normal_pair();
                [n1 * temp_scale, n2 * temp_scale]
            })
            .collect();

        // Run flow net: conditioning is already on GPU (bb_hidden)
        self.flow_x.upload(&noise);

        self.flow_ctx
            .set_input_shape("conditioning", &[1, D_MODEL as i64]);
        self.flow_ctx.set_input_shape("s", &[1, 1]);
        self.flow_ctx.set_input_shape("t", &[1, 1]);
        self.flow_ctx
            .set_input_shape("x", &[1, LATENT_DIM as i64]);

        self.flow_ctx
            .set_tensor_address("conditioning", self.bb_hidden.ptr);
        self.flow_ctx.set_tensor_address("s", self.flow_s.ptr);
        self.flow_ctx.set_tensor_address("t", self.flow_t.ptr);
        self.flow_ctx.set_tensor_address("x", self.flow_x.ptr);
        self.flow_ctx.set_tensor_address("v", self.flow_v.ptr);

        self.flow_ctx.enqueue(self.stream);
        unsafe { cudaStreamSynchronize(self.stream) };

        // Download velocity and compute latent = noise + velocity
        let mut velocity = vec![0f32; LATENT_DIM];
        self.flow_v.download(&mut velocity);

        let latent: Vec<f32> = noise
            .iter()
            .zip(velocity.iter())
            .map(|(&n, &v)| n + v)
            .collect();

        self.prev_latent = Some(latent.clone());
        self.latents.push(latent.clone());

        (latent, is_eos)
    }

    /// Batch-decode all accumulated latents to audio.
    pub fn decode_audio(&mut self) -> Vec<i16> {
        let num_frames = self.latents.len();
        if num_frames == 0 {
            return Vec::new();
        }

        // De-normalize latents and build tensor [1, LDIM, num_frames] (row-major)
        let mut latent_data = vec![0f32; LATENT_DIM * num_frames];
        for (frame_idx, latent) in self.latents.iter().enumerate() {
            let denorm = self.meta.denormalize_latent(latent);
            for (dim_idx, &val) in denorm.iter().enumerate() {
                latent_data[dim_idx * num_frames + frame_idx] = val;
            }
        }

        self.mimi_latent.upload(&latent_data);

        self.mimi_ctx
            .set_input_shape("latent", &[1, LATENT_DIM as i64, num_frames as i64]);
        self.mimi_ctx
            .set_tensor_address("latent", self.mimi_latent.ptr);
        self.mimi_ctx
            .set_tensor_address("audio", self.mimi_audio.ptr);

        self.mimi_ctx.enqueue(self.stream);
        unsafe { cudaStreamSynchronize(self.stream) };

        let est_samples = num_frames * 1920;
        let mut audio_f32 = vec![0f32; est_samples];
        self.mimi_audio.download(&mut audio_f32);

        audio_f32
            .iter()
            .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect()
    }
}

impl Drop for TrtPocket {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe { cudaStreamDestroy(self.stream) };
        }
    }
}

// ── Xorshift64 PRNG with Box-Muller normal generation ───────────────────

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
