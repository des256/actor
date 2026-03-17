use std::ffi::{c_char, c_void};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum TrtStatus {
    Ok = 0,
    Error = 1,
}

#[repr(C)]
pub struct TrtRuntime {
    _private: [u8; 0],
}

#[repr(C)]
pub struct TrtEngine {
    _private: [u8; 0],
}

#[repr(C)]
pub struct TrtContext {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn trt_get_last_error() -> *const c_char;
    pub fn trt_runtime_create(out: *mut *mut TrtRuntime) -> TrtStatus;
    pub fn trt_runtime_destroy(runtime: *mut TrtRuntime);
    pub fn trt_engine_load(runtime: *mut TrtRuntime, path: *const c_char, out: *mut *mut TrtEngine) -> TrtStatus;
    pub fn trt_engine_get_num_io_tensors(engine: *mut TrtEngine) -> i32;
    pub fn trt_engine_get_io_tensor_name(engine: *mut TrtEngine, index: i32) -> *const c_char;
    pub fn trt_engine_get_tensor_io_mode(engine: *mut TrtEngine, name: *const c_char) -> i32;
    pub fn trt_engine_get_tensor_dtype(engine: *mut TrtEngine, name: *const c_char) -> i32;
    pub fn trt_engine_get_tensor_shape(engine: *mut TrtEngine, name: *const c_char, dims: *mut i64, capacity: i32) -> i32;
    pub fn trt_engine_destroy(engine: *mut TrtEngine);
    pub fn trt_context_create(engine: *mut TrtEngine, out: *mut *mut TrtContext) -> TrtStatus;
    pub fn trt_context_set_input_shape(
        context: *mut TrtContext,
        name: *const c_char,
        dims: *const i64,
        ndims: i32,
    ) -> TrtStatus;
    pub fn trt_context_set_tensor_address(context: *mut TrtContext, name: *const c_char, ptr: *mut c_void) -> TrtStatus;
    pub fn trt_context_enqueue(context: *mut TrtContext, stream: *mut c_void) -> TrtStatus;
    pub fn trt_context_destroy(context: *mut TrtContext);
}
