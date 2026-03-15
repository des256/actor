use {
    crate::*,
    std::ffi::{CStr, CString},
};

/// Model architecture type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    DecoderOnly,
    EncoderOnly,
    EncoderDecoder,
}

impl ModelType {
    fn to_ffi(self) -> ffi::TrtLlmModelType {
        match self {
            ModelType::DecoderOnly => ffi::TrtLlmModelType::DecoderOnly,
            ModelType::EncoderOnly => ffi::TrtLlmModelType::EncoderOnly,
            ModelType::EncoderDecoder => ffi::TrtLlmModelType::EncoderDecoder,
        }
    }
}

/// Configuration for the TRT-LLM executor.
pub struct ExecutorConfig {
    pub model_type: ModelType,
    /// Fraction of free GPU memory to use for KV cache (0.0 = auto).
    pub kv_cache_free_gpu_mem_fraction: f32,
    /// Maximum beam width (1 = greedy decoding).
    pub max_beam_width: i32,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::DecoderOnly,
            kv_cache_free_gpu_mem_fraction: 0.0,
            max_beam_width: 1,
        }
    }
}

/// Sampling parameters for a single inference request.
pub struct SamplingParams {
    /// Sampling temperature (0.0 = use engine default).
    pub temperature: f32,
    /// Top-k sampling (0 = use engine default).
    pub top_k: i32,
    /// Top-p / nucleus sampling (0.0 = use engine default).
    pub top_p: f32,
    /// Repetition penalty (0.0 = use engine default).
    pub repetition_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 0.0,
            repetition_penalty: 0.0,
        }
    }
}

/// Safe wrapper around the TRT-LLM Executor.
pub struct Executor {
    handle: *mut ffi::TrtLlmExecutor,
}

// TRT-LLM's Executor is internally thread-safe.
unsafe impl Send for Executor {}
unsafe impl Sync for Executor {}

fn last_error() -> String {
    unsafe {
        let ptr = ffi::trtllm_get_last_error();
        if ptr.is_null() {
            "Unknown TRT-LLM error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

impl Executor {
    /// Create a new executor from an engine directory.
    pub fn new(engine_dir: &str, config: &ExecutorConfig) -> Self {
        let c_dir = CString::new(engine_dir).unwrap_or_else(|e| panic!("Null byte in engine_dir: {e}"));
        let mut handle: *mut ffi::TrtLlmExecutor = std::ptr::null_mut();
        let status = unsafe {
            ffi::trtllm_executor_create(
                c_dir.as_ptr(),
                config.model_type.to_ffi(),
                config.kv_cache_free_gpu_mem_fraction,
                config.max_beam_width,
                &mut handle,
            )
        };
        if status != ffi::TrtLlmStatus::Ok {
            panic!("Failed to create TRT-LLM executor: {}", last_error());
        }
        Self { handle }
    }

    /// Enqueue an inference request and return the request id.
    pub fn enqueue(
        &self,
        tokens: &[i32],
        max_new_tokens: i32,
        sampling: &SamplingParams,
        end_id: Option<i32>,
        pad_id: Option<i32>,
        streaming: bool,
    ) -> u64 {
        let mut req_id: u64 = 0;
        let status = unsafe {
            ffi::trtllm_executor_enqueue(
                self.handle,
                tokens.as_ptr(),
                tokens.len(),
                max_new_tokens,
                sampling.temperature,
                sampling.top_k,
                sampling.top_p,
                sampling.repetition_penalty,
                end_id.unwrap_or(-1),
                pad_id.unwrap_or(-1),
                streaming as i32,
                &mut req_id,
            )
        };
        if status != ffi::TrtLlmStatus::Ok {
            panic!("Failed to enqueue TRT-LLM request: {}", last_error());
        }
        req_id
    }

    /// Await a response for a request.
    ///
    /// Returns `(output_tokens, is_final)`. When streaming, call repeatedly
    /// until `is_final` is true. `timeout_ms` of 0 means wait indefinitely.
    pub fn await_response(&self, req_id: u64, timeout_ms: i64) -> (Vec<i32>, bool) {
        let mut buf = vec![0i32; 8192];
        let mut n_tokens: usize = 0;
        let mut is_final: i32 = 0;
        let status = unsafe {
            ffi::trtllm_executor_await(
                self.handle,
                req_id,
                buf.as_mut_ptr(),
                buf.len(),
                &mut n_tokens,
                &mut is_final,
                timeout_ms,
            )
        };
        if status != ffi::TrtLlmStatus::Ok {
            panic!("Failed to await TRT-LLM response for request {req_id}: {}", last_error());
        }
        buf.truncate(n_tokens);
        (buf, is_final != 0)
    }

    /// Cancel a pending request.
    pub fn cancel(&self, req_id: u64) {
        let status = unsafe { ffi::trtllm_executor_cancel(self.handle, req_id) };
        if status != ffi::TrtLlmStatus::Ok {
            panic!("Failed to cancel TRT-LLM request {req_id}: {}", last_error());
        }
    }

    /// Gracefully shut down the executor (blocks until all requests finish).
    pub fn shutdown(&self) {
        let status = unsafe { ffi::trtllm_executor_shutdown(self.handle) };
        if status != ffi::TrtLlmStatus::Ok {
            panic!("Failed to shut down TRT-LLM executor: {}", last_error());
        }
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::trtllm_executor_destroy(self.handle) };
        }
    }
}
