use std::{
    collections::HashMap,
    ffi::{CStr, CString, c_char, c_void},
    path::Path,
    ptr::null_mut,
    sync::Arc,
};

pub(crate) fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half >> 15) & 1) as u32;
    let exponent = ((half >> 10) & 0x1f) as u32;
    let mantissa = (half & 0x3ff) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            // ±0
            f32::from_bits(sign << 31)
        } else {
            // Denormalized: shift mantissa until hidden bit appears
            let mut e = 0i32;
            let mut m = mantissa;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let f32_exp = (127 - 15 + 1 + e) as u32;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exponent == 31 {
        // Infinity or NaN
        f32::from_bits((sign << 31) | (0xff << 23) | (mantissa << 13))
    } else {
        // Normalized
        let f32_exp = (exponent as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
    }
}

#[repr(C)]
pub struct OrtEnv {
    _private: [u8; 0],
}

unsafe impl Send for OrtEnv {}
unsafe impl Sync for OrtEnv {}

#[repr(C)]
pub struct OrtSession {
    _private: [u8; 0],
}

unsafe impl Send for OrtSession {}
unsafe impl Sync for OrtSession {}

#[repr(C)]
pub struct OrtSessionOptions {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OrtValue {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OrtStatus {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OrtMemoryInfo {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OrtAllocator {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OrtRunOptions {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OrtTensorTypeAndShapeInfo {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OrtTypeInfo {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OrtModelMetadata {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrtLoggingLevel {
    Verbose = 0,
    Info = 1,
    Warning = 2,
    Error = 3,
    Fatal = 4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ONNXTensorElementDataType {
    Undefined = 0,
    Float = 1,  // f32
    Uint8 = 2,  // u8
    Int8 = 3,   // i8
    Uint16 = 4, // u16
    Int16 = 5,  // i16
    Int32 = 6,  // i32
    Int64 = 7,  // i64
    String = 8,
    Bool = 9,
    Float16 = 10,
    Double = 11, // f64
    Uint32 = 12, // u32
    Uint64 = 13, // u64
    Complex64 = 14,
    Complex128 = 15,
    BFloat16 = 16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrtErrorCode {
    Ok = 0,
    Fail = 1,
    InvalidArgument = 2,
    NoSuchFile = 3,
    NoModel = 4,
    EngineError = 5,
    RuntimeException = 6,
    InvalidProtobuf = 7,
    ModelLoaded = 8,
    NotImplemented = 9,
    InvalidGraph = 10,
    EpFail = 11,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphOptimizationLevel {
    DisableAll = 0,
    EnableBasic = 1,
    EnableExtended = 2,
    EnableAll = 99,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrtMemType {
    CpuInput = -2,
    CpuOutput = -1,
    Default = 0,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrtAllocatorType {
    Invalid = -1,
    Device = 0,
    Arena = 1,
}

#[repr(C)]
#[allow(non_snake_case)]
pub struct OrtApiBase {
    pub GetApi: unsafe extern "C" fn(version: u32) -> *const OrtApi,
    pub GetVersionString: unsafe extern "C" fn() -> *const c_char,
}

unsafe impl Send for OrtApiBase {}

#[repr(C)]
pub struct OrtApi {
    _private: [u8; 0],
}

unsafe impl Send for OrtApi {}
unsafe impl Sync for OrtApi {}

unsafe extern "C" {
    pub fn OrtGetApiBase() -> *const OrtApiBase;
}

unsafe extern "C" {
    pub fn OrtSessionOptionsAppendExecutionProvider_CUDA(options: *mut OrtSessionOptions, device_id: i32) -> *mut OrtStatus;
}

pub type CreateStatusFn = unsafe extern "C" fn(code: OrtErrorCode, msg: *const c_char) -> *mut OrtStatus;
pub type GetErrorCodeFn = unsafe extern "C" fn(status: *const OrtStatus) -> OrtErrorCode;
pub type GetErrorMessageFn = unsafe extern "C" fn(status: *const OrtStatus) -> *const c_char;
pub type CreateEnvFn =
    unsafe extern "C" fn(log_level: OrtLoggingLevel, log_id: *const c_char, out: *mut *mut OrtEnv) -> *mut OrtStatus;
pub type ReleaseEnvFn = unsafe extern "C" fn(env: *mut OrtEnv);
pub type GetAllocatorWithDefaultOptionsFn = unsafe extern "C" fn(out: *mut *mut OrtAllocator) -> *mut OrtStatus;
pub type AllocatorFreeFn = unsafe extern "C" fn(allocator: *mut OrtAllocator, ptr: *mut c_void);
pub type ReleaseStatusFn = unsafe extern "C" fn(status: *mut OrtStatus);
pub type CreateSessionFn = unsafe extern "C" fn(
    env: *const OrtEnv,
    model_path: *const c_char,
    options: *const OrtSessionOptions,
    out: *mut *mut OrtSession,
) -> *mut OrtStatus;
pub type ReleaseSessionFn = unsafe extern "C" fn(session: *mut OrtSession);
pub type RunFn = unsafe extern "C" fn(
    session: *mut OrtSession,
    run_options: *const OrtRunOptions,
    input_names: *const *const c_char,
    inputs: *const *const OrtValue,
    input_len: usize,
    output_names: *const *const c_char,
    output_names_len: usize,
    outputs: *mut *mut OrtValue,
) -> *mut OrtStatus;

pub type SessionGetInputCountFn = unsafe extern "C" fn(session: *const OrtSession, out: *mut usize) -> *mut OrtStatus;

pub type SessionGetOutputCountFn = unsafe extern "C" fn(session: *const OrtSession, out: *mut usize) -> *mut OrtStatus;

pub type SessionGetInputNameFn = unsafe extern "C" fn(
    session: *const OrtSession,
    index: usize,
    allocator: *mut OrtAllocator,
    value: *mut *mut c_char,
) -> *mut OrtStatus;

pub type SessionGetOutputNameFn = unsafe extern "C" fn(
    session: *const OrtSession,
    index: usize,
    allocator: *mut OrtAllocator,
    value: *mut *mut c_char,
) -> *mut OrtStatus;

pub type SessionGetInputTypeInfoFn =
    unsafe extern "C" fn(session: *const OrtSession, index: usize, type_info: *mut *mut OrtTypeInfo) -> *mut OrtStatus;

pub type CreateSessionOptionsFn = unsafe extern "C" fn(out: *mut *mut OrtSessionOptions) -> *mut OrtStatus;
pub type ReleaseSessionOptionsFn = unsafe extern "C" fn(options: *mut OrtSessionOptions);
pub type SetSessionGraphOptimizationLevelFn =
    unsafe extern "C" fn(options: *mut OrtSessionOptions, level: GraphOptimizationLevel) -> *mut OrtStatus;
pub type SetIntraOpNumThreadsFn = unsafe extern "C" fn(options: *mut OrtSessionOptions, num_threads: i32) -> *mut OrtStatus;
pub type CreateCpuMemoryInfoFn = unsafe extern "C" fn(
    allocator_type: OrtAllocatorType,
    mem_type: OrtMemType,
    out: *mut *mut OrtMemoryInfo,
) -> *mut OrtStatus;
pub type ReleaseMemoryInfoFn = unsafe extern "C" fn(info: *mut OrtMemoryInfo);
pub type CreateTensorWithDataAsOrtValueFn = unsafe extern "C" fn(
    memory_info: *const OrtMemoryInfo,
    data: *mut c_void,
    data_len: usize,
    shape: *const i64,
    shape_len: usize,
    element_type: ONNXTensorElementDataType,
    out: *mut *mut OrtValue,
) -> *mut OrtStatus;
pub type ReleaseValueFn = unsafe extern "C" fn(value: *mut OrtValue);
pub type GetTensorMutableDataFn = unsafe extern "C" fn(value: *mut OrtValue, out: *mut *mut c_void) -> *mut OrtStatus;
pub type GetTensorTypeAndShapeFn =
    unsafe extern "C" fn(value: *const OrtValue, out: *mut *mut OrtTensorTypeAndShapeInfo) -> *mut OrtStatus;
pub type ReleaseTensorTypeAndShapeInfoFn = unsafe extern "C" fn(info: *mut OrtTensorTypeAndShapeInfo);
pub type ReleaseTypeInfoFn = unsafe extern "C" fn(info: *mut OrtTypeInfo);
pub type GetTensorElementTypeFn =
    unsafe extern "C" fn(info: *const OrtTensorTypeAndShapeInfo, out: *mut ONNXTensorElementDataType) -> *mut OrtStatus;
pub type CastTypeInfoToTensorInfoFn =
    unsafe extern "C" fn(type_info: *const OrtTypeInfo, out: *mut *const OrtTensorTypeAndShapeInfo) -> *mut OrtStatus;
pub type GetDimensionsCountFn = unsafe extern "C" fn(info: *const OrtTensorTypeAndShapeInfo, out: *mut usize) -> *mut OrtStatus;
pub type GetDimensionsFn =
    unsafe extern "C" fn(info: *const OrtTensorTypeAndShapeInfo, dim_values: *mut i64, dim_count: usize) -> *mut OrtStatus;
pub type GetTensorShapeElementCountFn =
    unsafe extern "C" fn(info: *const OrtTensorTypeAndShapeInfo, out: *mut usize) -> *mut OrtStatus;
pub type SessionGetModelMetadataFn =
    unsafe extern "C" fn(session: *const OrtSession, out: *mut *mut OrtModelMetadata) -> *mut OrtStatus;
pub type ModelMetadataLookupCustomMetadataMapFn = unsafe extern "C" fn(
    model_metadata: *const OrtModelMetadata,
    allocator: *mut OrtAllocator,
    key: *const c_char,
    value: *mut *mut c_char,
) -> *mut OrtStatus;
pub type ModelMetadataGetCustomMetadataMapKeysFn = unsafe extern "C" fn(
    model_metadata: *const OrtModelMetadata,
    allocator: *mut OrtAllocator,
    keys: *mut *mut *mut c_char,
    num_keys: *mut i64,
) -> *mut OrtStatus;
pub type ReleaseModelMetadataFn = unsafe extern "C" fn(metadata: *mut OrtModelMetadata);

pub const IDX_CREATE_STATUS: usize = 0;
pub const IDX_GET_ERROR_CODE: usize = 1;
pub const IDX_GET_ERROR_MESSAGE: usize = 2;
pub const IDX_CREATE_ENV: usize = 3;
pub const IDX_CREATE_SESSION: usize = 7;
pub const IDX_RUN: usize = 9;
pub const IDX_CREATE_SESSION_OPTIONS: usize = 10;
pub const IDX_SESSION_GET_INPUT_COUNT: usize = 30;
pub const IDX_SESSION_GET_OUTPUT_COUNT: usize = 31;
pub const IDX_SESSION_GET_INPUT_TYPE_INFO: usize = 33;
pub const IDX_SESSION_GET_INPUT_NAME: usize = 36;
pub const IDX_SESSION_GET_OUTPUT_NAME: usize = 37;
pub const IDX_ALLOCATOR_FREE: usize = 76;
pub const IDX_GET_ALLOCATOR_WITH_DEFAULT_OPTIONS: usize = 78;
pub const IDX_SET_SESSION_GRAPH_OPTIMIZATION_LEVEL: usize = 23;
pub const IDX_SET_INTRA_OP_NUM_THREADS: usize = 24;
pub const IDX_CREATE_TENSOR_WITH_DATA_AS_ORT_VALUE: usize = 49;
pub const IDX_GET_TENSOR_MUTABLE_DATA: usize = 51;
pub const IDX_CAST_TYPE_INFO_TO_TENSOR_INFO: usize = 55;
pub const IDX_GET_DIMENSIONS_COUNT: usize = 61;
pub const IDX_GET_DIMENSIONS: usize = 62;
pub const IDX_GET_TENSOR_ELEMENT_TYPE: usize = 60;
pub const IDX_GET_TENSOR_SHAPE_ELEMENT_COUNT: usize = 64;
pub const IDX_GET_TENSOR_TYPE_AND_SHAPE: usize = 65;
pub const IDX_CREATE_CPU_MEMORY_INFO: usize = 69;
pub const IDX_RELEASE_ENV: usize = 92;
pub const IDX_RELEASE_STATUS: usize = 93;
pub const IDX_RELEASE_MEMORY_INFO: usize = 94;
pub const IDX_RELEASE_SESSION: usize = 95;
pub const IDX_RELEASE_VALUE: usize = 96;
pub const IDX_RELEASE_RUN_OPTIONS: usize = 97;
pub const IDX_RELEASE_TYPE_INFO: usize = 98;
pub const IDX_RELEASE_TENSOR_TYPE_AND_SHAPE_INFO: usize = 99;
pub const IDX_RELEASE_SESSION_OPTIONS: usize = 100;
pub const IDX_SESSION_GET_MODEL_METADATA: usize = 111;
pub const IDX_MODEL_METADATA_LOOKUP_CUSTOM_METADATA_MAP: usize = 116;
pub const IDX_RELEASE_MODEL_METADATA: usize = 118;
pub const IDX_MODEL_METADATA_GET_CUSTOM_METADATA_MAP_KEYS: usize = 123;

impl OrtApi {
    pub unsafe fn get_fn<F>(&self, index: usize) -> F {
        unsafe {
            let vtable = self as *const _ as *const *const ();
            let fn_ptr = *vtable.add(index);
            std::mem::transmute_copy(&fn_ptr)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opaque_types_are_zero_sized() {
        // Opaque types should be zero-sized for correct FFI
        assert_eq!(std::mem::size_of::<OrtEnv>(), 0);
        assert_eq!(std::mem::size_of::<OrtSession>(), 0);
        assert_eq!(std::mem::size_of::<OrtSessionOptions>(), 0);
        assert_eq!(std::mem::size_of::<OrtValue>(), 0);
        assert_eq!(std::mem::size_of::<OrtStatus>(), 0);
    }

    #[test]
    fn test_enum_discriminants() {
        // Verify enum values match the C API
        assert_eq!(OrtLoggingLevel::Warning as i32, 2);
        assert_eq!(OrtErrorCode::Ok as i32, 0);
        assert_eq!(OrtErrorCode::Fail as i32, 1);
        assert_eq!(ONNXTensorElementDataType::Float as i32, 1);
        assert_eq!(ONNXTensorElementDataType::Int64 as i32, 7);
        assert_eq!(ONNXTensorElementDataType::Double as i32, 11);
        assert_eq!(GraphOptimizationLevel::EnableAll as i32, 99);
    }
}

#[derive(Clone, Copy)]
pub enum Executor {
    Cpu,
    Cuda(usize),
}

#[derive(Clone, Copy)]
pub enum OptimizationLevel {
    Disabled,
    EnableBasic,
    EnableExtended,
    EnableAll,
}

pub struct Onnx {
    pub(crate) api: *const OrtApi,
    pub(crate) environment: *mut OrtEnv,
    pub(crate) allocator: *mut OrtAllocator,
    pub(crate) get_error_code: GetErrorCodeFn,
    pub(crate) get_error_message: GetErrorMessageFn,
    pub(crate) allocator_free: AllocatorFreeFn,
    pub(crate) create_session: CreateSessionFn,
    pub(crate) create_session_options: CreateSessionOptionsFn,
    pub(crate) set_session_graph_optimization_level: SetSessionGraphOptimizationLevelFn,
    pub(crate) set_intra_op_num_threads: SetIntraOpNumThreadsFn,
    pub(crate) release_session_options: ReleaseSessionOptionsFn,
    pub(crate) session_get_input_count: SessionGetInputCountFn,
    pub(crate) session_get_output_count: SessionGetOutputCountFn,
    pub(crate) session_get_input_name: SessionGetInputNameFn,
    pub(crate) session_get_output_name: SessionGetOutputNameFn,
    pub(crate) session_get_input_type_info: SessionGetInputTypeInfoFn,
    pub(crate) release_type_info: ReleaseTypeInfoFn,
    pub(crate) cast_type_info_to_tensor_info: CastTypeInfoToTensorInfoFn,
    pub(crate) get_dimensions_count: GetDimensionsCountFn,
    pub(crate) get_dimensions: GetDimensionsFn,
    pub(crate) get_tensor_element_type: GetTensorElementTypeFn,
    pub(crate) run: RunFn,
    pub(crate) release_value: ReleaseValueFn,
    pub(crate) release_session: ReleaseSessionFn,
    pub(crate) create_memory_info: CreateCpuMemoryInfoFn,
    pub(crate) create_tensor: CreateTensorWithDataAsOrtValueFn,
    pub(crate) release_memory_info: ReleaseMemoryInfoFn,
    pub(crate) get_tensor_type_and_shape: GetTensorTypeAndShapeFn,
    pub(crate) get_tensor_shape_element_count: GetTensorShapeElementCountFn,
    pub(crate) release_tensor_type_and_shape_info: ReleaseTensorTypeAndShapeInfoFn,
    pub(crate) get_tensor_mutable_data: GetTensorMutableDataFn,
    pub(crate) session_get_model_metadata: SessionGetModelMetadataFn,
    pub(crate) model_metadata_lookup_custom_metadata_map: ModelMetadataLookupCustomMetadataMapFn,
    pub(crate) model_metadata_get_custom_metadata_map_keys: ModelMetadataGetCustomMetadataMapKeysFn,
    pub(crate) release_model_metadata: ReleaseModelMetadataFn,
}

unsafe impl Send for Onnx {}
unsafe impl Sync for Onnx {}

impl Onnx {
    pub(crate) fn status_to_string(&self, status: *mut OrtStatus) -> String {
        let code = unsafe { (self.get_error_code)(status) };
        let msg_ptr = unsafe { (self.get_error_message)(status) };
        let message = if msg_ptr.is_null() {
            String::from("Unknown error")
        } else {
            unsafe { CStr::from_ptr(msg_ptr).to_string_lossy().into_owned() }
        };
        let release_status: ReleaseStatusFn = unsafe { (*self.api).get_fn(IDX_RELEASE_STATUS) };
        unsafe { release_status(status) };
        format!("{:?}: {}", code, message)
    }

    pub fn new(version: usize) -> Arc<Self> {
        let api_base = unsafe { OrtGetApiBase() };
        if api_base.is_null() {
            panic!("Failed to get ONNX runtime API base");
        }
        let get_api = unsafe { (*api_base).GetApi };
        let api = unsafe { get_api(version as u32) };
        if api.is_null() {
            panic!("ONNX runtime doesn't support API version {}", version);
        }
        // get function pointers
        let create_env: CreateEnvFn = unsafe { (*api).get_fn(IDX_CREATE_ENV) };
        let get_error_code: GetErrorCodeFn = unsafe { (*api).get_fn(IDX_GET_ERROR_CODE) };
        let get_error_message: GetErrorMessageFn = unsafe { (*api).get_fn(IDX_GET_ERROR_MESSAGE) };
        let allocator_free: AllocatorFreeFn = unsafe { (*api).get_fn(IDX_ALLOCATOR_FREE) };
        let create_session: CreateSessionFn = unsafe { (*api).get_fn(IDX_CREATE_SESSION) };
        let create_session_options: CreateSessionOptionsFn = unsafe { (*api).get_fn(IDX_CREATE_SESSION_OPTIONS) };
        let set_session_graph_optimization_level: SetSessionGraphOptimizationLevelFn =
            unsafe { (*api).get_fn(IDX_SET_SESSION_GRAPH_OPTIMIZATION_LEVEL) };
        let set_intra_op_num_threads: SetIntraOpNumThreadsFn = unsafe { (*api).get_fn(IDX_SET_INTRA_OP_NUM_THREADS) };
        let release_session_options: ReleaseSessionOptionsFn = unsafe { (*api).get_fn(IDX_RELEASE_SESSION_OPTIONS) };
        let session_get_input_count: SessionGetInputCountFn = unsafe { (*api).get_fn(IDX_SESSION_GET_INPUT_COUNT) };
        let session_get_output_count: SessionGetOutputCountFn = unsafe { (*api).get_fn(IDX_SESSION_GET_OUTPUT_COUNT) };
        let session_get_input_name: SessionGetInputNameFn = unsafe { (*api).get_fn(IDX_SESSION_GET_INPUT_NAME) };
        let session_get_output_name: SessionGetOutputNameFn = unsafe { (*api).get_fn(IDX_SESSION_GET_OUTPUT_NAME) };
        let session_get_input_type_info: SessionGetInputTypeInfoFn = unsafe { (*api).get_fn(IDX_SESSION_GET_INPUT_TYPE_INFO) };
        let release_type_info: ReleaseTypeInfoFn = unsafe { (*api).get_fn(IDX_RELEASE_TYPE_INFO) };
        let cast_type_info_to_tensor_info: CastTypeInfoToTensorInfoFn =
            unsafe { (*api).get_fn(IDX_CAST_TYPE_INFO_TO_TENSOR_INFO) };
        let get_dimensions_count: GetDimensionsCountFn = unsafe { (*api).get_fn(IDX_GET_DIMENSIONS_COUNT) };
        let get_dimensions: GetDimensionsFn = unsafe { (*api).get_fn(IDX_GET_DIMENSIONS) };
        let get_tensor_element_type: GetTensorElementTypeFn = unsafe { (*api).get_fn(IDX_GET_TENSOR_ELEMENT_TYPE) };
        let run: RunFn = unsafe { (*api).get_fn(IDX_RUN) };
        let release_value: ReleaseValueFn = unsafe { (*api).get_fn(IDX_RELEASE_VALUE) };
        let release_session: ReleaseSessionFn = unsafe { (*api).get_fn(IDX_RELEASE_SESSION) };
        let create_memory_info: CreateCpuMemoryInfoFn = unsafe { (*api).get_fn(IDX_CREATE_CPU_MEMORY_INFO) };
        let create_tensor: CreateTensorWithDataAsOrtValueFn =
            unsafe { (*api).get_fn(IDX_CREATE_TENSOR_WITH_DATA_AS_ORT_VALUE) };
        let release_memory_info: ReleaseMemoryInfoFn = unsafe { (*api).get_fn(IDX_RELEASE_MEMORY_INFO) };
        let get_tensor_type_and_shape: GetTensorTypeAndShapeFn = unsafe { (*api).get_fn(IDX_GET_TENSOR_TYPE_AND_SHAPE) };
        let get_tensor_shape_element_count: GetTensorShapeElementCountFn =
            unsafe { (*api).get_fn(IDX_GET_TENSOR_SHAPE_ELEMENT_COUNT) };
        let release_tensor_type_and_shape_info: ReleaseTensorTypeAndShapeInfoFn =
            unsafe { (*api).get_fn(IDX_RELEASE_TENSOR_TYPE_AND_SHAPE_INFO) };
        let get_tensor_mutable_data: GetTensorMutableDataFn = unsafe { (*api).get_fn(IDX_GET_TENSOR_MUTABLE_DATA) };
        let session_get_model_metadata: SessionGetModelMetadataFn = unsafe { (*api).get_fn(IDX_SESSION_GET_MODEL_METADATA) };
        let model_metadata_lookup_custom_metadata_map: ModelMetadataLookupCustomMetadataMapFn =
            unsafe { (*api).get_fn(IDX_MODEL_METADATA_LOOKUP_CUSTOM_METADATA_MAP) };
        let model_metadata_get_custom_metadata_map_keys: ModelMetadataGetCustomMetadataMapKeysFn =
            unsafe { (*api).get_fn(IDX_MODEL_METADATA_GET_CUSTOM_METADATA_MAP_KEYS) };
        let release_model_metadata: ReleaseModelMetadataFn = unsafe { (*api).get_fn(IDX_RELEASE_MODEL_METADATA) };
        let log_id = CString::new("onnx").unwrap();
        let mut environment: *mut OrtEnv = null_mut();
        let status = unsafe { create_env(OrtLoggingLevel::Warning, log_id.as_ptr(), &mut environment as *mut _) };
        if !status.is_null() {
            panic!("Failed to create ONNX runtime environment");
        }
        let mut allocator: *mut OrtAllocator = std::ptr::null_mut();
        let get_allocator_with_default_options: GetAllocatorWithDefaultOptionsFn =
            unsafe { (*api).get_fn(IDX_GET_ALLOCATOR_WITH_DEFAULT_OPTIONS) };
        let status = unsafe { get_allocator_with_default_options(&mut allocator as *mut _) };
        if !status.is_null() {
            panic!("Failed to get allocator with default options");
        }
        Arc::new(Self {
            api,
            environment,
            allocator,
            get_error_code,
            get_error_message,
            allocator_free,
            create_session,
            create_session_options,
            set_session_graph_optimization_level,
            set_intra_op_num_threads,
            release_session_options,
            session_get_input_count,
            session_get_output_count,
            session_get_input_name,
            session_get_output_name,
            session_get_input_type_info,
            release_type_info,
            cast_type_info_to_tensor_info,
            get_dimensions_count,
            get_dimensions,
            get_tensor_element_type,
            run,
            release_value,
            release_session,
            create_memory_info,
            create_tensor,
            release_memory_info,
            get_tensor_type_and_shape,
            get_tensor_shape_element_count,
            release_tensor_type_and_shape_info,
            get_tensor_mutable_data,
            session_get_model_metadata,
            model_metadata_lookup_custom_metadata_map,
            model_metadata_get_custom_metadata_map_keys,
            release_model_metadata,
        })
    }

    pub fn create_session(
        self: &Arc<Self>,
        executor: Executor,
        optimization_level: OptimizationLevel,
        threads: usize,
        model_path: impl AsRef<Path>,
    ) -> Session {
        let mut options: *mut OrtSessionOptions = null_mut();
        let status = unsafe { (self.create_session_options)(&mut options as *mut _) };
        if !status.is_null() {
            panic!("Failed to create session options: {}", self.status_to_string(status));
        }
        if let Executor::Cuda(id) = executor {
            let status = unsafe { OrtSessionOptionsAppendExecutionProvider_CUDA(options, id as i32) };
            if !status.is_null() {
                unsafe { (self.release_session_options)(options) };
                panic!("Failed to append execution provider CUDA: {}", self.status_to_string(status));
            }
        }
        let status = unsafe {
            (self.set_session_graph_optimization_level)(
                options,
                match optimization_level {
                    OptimizationLevel::Disabled => GraphOptimizationLevel::DisableAll,
                    OptimizationLevel::EnableBasic => GraphOptimizationLevel::EnableBasic,
                    OptimizationLevel::EnableExtended => GraphOptimizationLevel::EnableExtended,
                    OptimizationLevel::EnableAll => GraphOptimizationLevel::EnableAll,
                },
            )
        };
        if !status.is_null() {
            unsafe { (self.release_session_options)(options) };
            panic!(
                "Failed to set session graph optimization level: {}",
                self.status_to_string(status)
            );
        }
        let status = unsafe { (self.set_intra_op_num_threads)(options, threads as i32) };
        if !status.is_null() {
            unsafe { (self.release_session_options)(options) };
            panic!("Failed to set intra-op num threads: {}", self.status_to_string(status));
        }
        let path_str = model_path.as_ref().to_str().unwrap();
        let c_path: CString = match CString::new(path_str) {
            Ok(c_path) => c_path,
            Err(error) => panic!("Null byte in model path: {}", error),
        };
        let mut session: *mut OrtSession = null_mut();
        let status = unsafe { (self.create_session)(self.environment, c_path.as_ptr(), options, &mut session as *mut _) };
        if !status.is_null() {
            unsafe { (self.release_session_options)(options) };
            panic!("Failed to create session: {}", self.status_to_string(status));
        }
        unsafe { (self.release_session_options)(options) };
        Session {
            onnx: Arc::clone(&self),
            session,
        }
    }
}

pub struct Session {
    pub(crate) onnx: Arc<Onnx>,
    pub(crate) session: *mut OrtSession,
}

unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl Session {
    pub fn input_count(&self) -> usize {
        let mut count: usize = 0;
        let status = unsafe { (self.onnx.session_get_input_count)(self.session, &mut count as *mut _) };
        if !status.is_null() {
            panic!("Failed to get input count: {}", self.onnx.status_to_string(status));
        }
        count
    }

    pub fn output_count(&self) -> usize {
        let mut count: usize = 0;
        let status = unsafe { (self.onnx.session_get_output_count)(self.session, &mut count as *mut _) };
        if !status.is_null() {
            panic!("Failed to get output count: {}", self.onnx.status_to_string(status));
        }
        count
    }

    pub fn input_name(&self, index: usize) -> String {
        let mut name_ptr: *mut c_char = null_mut();
        let status =
            unsafe { (self.onnx.session_get_input_name)(self.session, index, self.onnx.allocator, &mut name_ptr as *mut _) };
        if !status.is_null() {
            unsafe { (self.onnx.allocator_free)(self.onnx.allocator, name_ptr as *mut c_void) };
            panic!("Failed to get input name: {}", self.onnx.status_to_string(status));
        }

        let name = unsafe {
            CStr::from_ptr(name_ptr)
                .to_str()
                .map_err(|_| panic!("Invalid UTF-8 in input name"))
                .unwrap()
        };
        unsafe { (self.onnx.allocator_free)(self.onnx.allocator, name_ptr as *mut c_void) };
        name.to_string()
    }

    pub fn output_name(&self, index: usize) -> String {
        let mut name_ptr: *mut c_char = null_mut();
        let status =
            unsafe { (self.onnx.session_get_output_name)(self.session, index, self.onnx.allocator, &mut name_ptr as *mut _) };
        if !status.is_null() {
            unsafe { (self.onnx.allocator_free)(self.onnx.allocator, name_ptr as *mut c_void) };
            panic!("Failed to get output name: {}", self.onnx.status_to_string(status));
        }
        let name = unsafe {
            CStr::from_ptr(name_ptr)
                .to_str()
                .map_err(|_| panic!("Invalid UTF-8 in output name"))
                .unwrap()
        };
        unsafe { (self.onnx.allocator_free)(self.onnx.allocator, name_ptr as *mut c_void) };
        name.to_string()
    }

    fn get_type_info(&self, index: usize) -> *mut OrtTypeInfo {
        let mut type_info: *mut OrtTypeInfo = std::ptr::null_mut();
        let status = unsafe { (self.onnx.session_get_input_type_info)(self.session, index, &mut type_info as *mut _) };
        if !status.is_null() {
            panic!("Failed to get input type info: {}", self.onnx.status_to_string(status));
        }
        type_info
    }

    fn release_type_info(&self, type_info: *mut OrtTypeInfo) {
        unsafe { (self.onnx.release_type_info)(type_info) };
    }

    pub fn input_shape(&self, index: usize) -> Vec<i64> {
        let type_info = self.get_type_info(index);

        let mut tensor_info: *const OrtTensorTypeAndShapeInfo = null_mut();
        let status = unsafe { (self.onnx.cast_type_info_to_tensor_info)(type_info, &mut tensor_info as *mut _) };
        if !status.is_null() {
            self.release_type_info(type_info);
            panic!(
                "Failed to cast type info to tensor info: {}",
                self.onnx.status_to_string(status)
            );
        }
        if tensor_info.is_null() {
            self.release_type_info(type_info);
            panic!("Input is not a tensor type");
        }
        let mut dim_count: usize = 0;
        let status = unsafe { (self.onnx.get_dimensions_count)(tensor_info, &mut dim_count as *mut _) };
        if !status.is_null() {
            self.release_type_info(type_info);
            panic!("Failed to get dimensions count: {}", self.onnx.status_to_string(status));
        }
        let mut dims = vec![0i64; dim_count];
        let status = unsafe { (self.onnx.get_dimensions)(tensor_info, dims.as_mut_ptr(), dim_count) };
        if !status.is_null() {
            self.release_type_info(type_info);
            panic!("Failed to get dimensions count: {}", self.onnx.status_to_string(status));
        }
        self.release_type_info(type_info);
        dims
    }

    pub fn input_element_type(&self, index: usize) -> ONNXTensorElementDataType {
        let mut type_info: *mut OrtTypeInfo = null_mut();
        let status = unsafe { (self.onnx.session_get_input_type_info)(self.session, index, &mut type_info as *mut _) };
        if !status.is_null() {
            panic!("Failed to get input type info: {}", self.onnx.status_to_string(status));
        }
        let mut tensor_info: *const OrtTensorTypeAndShapeInfo = std::ptr::null();
        let status = unsafe { (self.onnx.cast_type_info_to_tensor_info)(type_info, &mut tensor_info as *mut _) };
        if !status.is_null() {
            self.release_type_info(type_info);
            panic!(
                "Failed to cast type info to tensor info: {}",
                self.onnx.status_to_string(status)
            );
        }
        let mut elem_type = ONNXTensorElementDataType::Undefined;
        let status = unsafe { (self.onnx.get_tensor_element_type)(tensor_info, &mut elem_type as *mut _) };
        self.release_type_info(type_info);
        if !status.is_null() {
            panic!("Failed to get tensor element type: {}", self.onnx.status_to_string(status));
        }
        elem_type
    }

    /// Get custom metadata from the model as a key-value map.
    pub fn metadata(&self) -> HashMap<String, String> {
        let mut metadata: *mut OrtModelMetadata = std::ptr::null_mut();
        let status = unsafe { (self.onnx.session_get_model_metadata)(self.session, &mut metadata as *mut _) };
        if !status.is_null() {
            panic!("Failed to get model metadata: {}", self.onnx.status_to_string(status));
        }
        let mut keys_ptr: *mut *mut c_char = null_mut();
        let mut num_keys: i64 = 0;
        let status = unsafe {
            (self.onnx.model_metadata_get_custom_metadata_map_keys)(
                metadata,
                self.onnx.allocator,
                &mut keys_ptr as *mut _,
                &mut num_keys as *mut _,
            )
        };
        if !status.is_null() {
            unsafe { (self.onnx.release_model_metadata)(metadata) };
            panic!("Failed to get model metadata keys: {}", self.onnx.status_to_string(status));
        }
        let mut map = HashMap::new();
        for i in 0..num_keys as usize {
            let key_ptr = unsafe { *keys_ptr.add(i) };
            let key = unsafe { CStr::from_ptr(key_ptr).to_string_lossy().into_owned() };
            let key_cstr: CString = match CString::new(key.as_str()) {
                Ok(c_str) => c_str,
                Err(error) => panic!("Null byte in metadata key: {}", error),
            };
            let mut value_ptr: *mut c_char = null_mut();
            let status = unsafe {
                (self.onnx.model_metadata_lookup_custom_metadata_map)(
                    metadata,
                    self.onnx.allocator,
                    key_cstr.as_ptr(),
                    &mut value_ptr as *mut _,
                )
            };
            if !status.is_null() {
                unsafe { (self.onnx.allocator_free)(self.onnx.allocator, key_ptr as *mut _) };
                for j in (i + 1)..num_keys as usize {
                    unsafe { (self.onnx.allocator_free)(self.onnx.allocator, *keys_ptr.add(j) as *mut _) };
                }
                unsafe { (self.onnx.allocator_free)(self.onnx.allocator, keys_ptr as *mut _) };
                unsafe { (self.onnx.release_model_metadata)(metadata) };
                panic!("Failed to lookup custom metadata map: {}", self.onnx.status_to_string(status));
            }
            if !value_ptr.is_null() {
                let value = unsafe { CStr::from_ptr(value_ptr).to_string_lossy().into_owned() };
                map.insert(key, value);
                unsafe { (self.onnx.allocator_free)(self.onnx.allocator, value_ptr as *mut _) };
            }
            unsafe { (self.onnx.allocator_free)(self.onnx.allocator, key_ptr as *mut _) };
        }
        if !keys_ptr.is_null() {
            unsafe { (self.onnx.allocator_free)(self.onnx.allocator, keys_ptr as *mut _) };
        }
        unsafe { (self.onnx.release_model_metadata)(metadata) };
        map
    }

    pub fn run(&mut self, inputs: &[(&str, &Value)], output_names: &[&str]) -> Vec<Value> {
        let input_name_cstrings: Vec<CString> = inputs
            .iter()
            .map(|(name, _)| match CString::new(*name) {
                Ok(c_str) => c_str,
                Err(error) => panic!("Null byte in input name: {}", error),
            })
            .collect();
        let input_name_ptrs: Vec<_> = input_name_cstrings.iter().map(|s| s.as_ptr()).collect();
        let input_value_ptrs: Vec<_> = inputs.iter().map(|(_, value)| value.as_ptr()).collect();
        let output_name_cstrings: Vec<CString> = output_names
            .iter()
            .map(|name| match CString::new(*name) {
                Ok(c_str) => c_str,
                Err(error) => panic!("Null byte in output name: {}", error),
            })
            .collect();
        let output_name_ptrs: Vec<_> = output_name_cstrings.iter().map(|s| s.as_ptr()).collect();
        let mut output_value_ptrs: Vec<*mut OrtValue> = vec![std::ptr::null_mut(); output_names.len()];
        let status = unsafe {
            (self.onnx.run)(
                self.session,
                std::ptr::null(), // run_options (null = default)
                input_name_ptrs.as_ptr(),
                input_value_ptrs.as_ptr(),
                inputs.len(),
                output_name_ptrs.as_ptr(),
                output_names.len(),
                output_value_ptrs.as_mut_ptr(),
            )
        };
        if !status.is_null() {
            for &output_ptr in &output_value_ptrs {
                if !output_ptr.is_null() {
                    unsafe { (self.onnx.release_value)(output_ptr) };
                }
            }
            panic!("Failed to run inference: {}", self.onnx.status_to_string(status));
        }
        let outputs: Vec<_> = output_value_ptrs
            .into_iter()
            .map(|value_ptr| unsafe { Value::from_raw(&self.onnx, value_ptr) })
            .collect();
        outputs
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe { (self.onnx.release_session)(self.session) };
    }
}

mod sealed {
    pub trait Sealed {}
}

pub trait TensorElement: sealed::Sealed + Sized + Copy {
    fn element_type() -> ONNXTensorElementDataType;
}

impl sealed::Sealed for f32 {}
impl TensorElement for f32 {
    fn element_type() -> ONNXTensorElementDataType {
        ONNXTensorElementDataType::Float
    }
}

impl sealed::Sealed for f64 {}
impl TensorElement for f64 {
    fn element_type() -> ONNXTensorElementDataType {
        ONNXTensorElementDataType::Double
    }
}

impl sealed::Sealed for i64 {}
impl TensorElement for i64 {
    fn element_type() -> ONNXTensorElementDataType {
        ONNXTensorElementDataType::Int64
    }
}

impl sealed::Sealed for i32 {}
impl TensorElement for i32 {
    fn element_type() -> ONNXTensorElementDataType {
        ONNXTensorElementDataType::Int32
    }
}

impl sealed::Sealed for bool {}
impl TensorElement for bool {
    fn element_type() -> ONNXTensorElementDataType {
        ONNXTensorElementDataType::Bool
    }
}

pub struct Value {
    onnx: Arc<Onnx>,
    value: *mut OrtValue,
    _data: Box<[u8]>,
}

unsafe impl Send for Value {}

impl Value {
    pub fn from_slice<T: TensorElement>(onnx: &Arc<Onnx>, shape: &[usize], data: &[T]) -> Self {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            panic!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape,
                expected_len
            );
        }
        let byte_len = data.len() * std::mem::size_of::<T>();
        let mut buffer = vec![0u8; byte_len].into_boxed_slice();
        let src_ptr = data.as_ptr() as *const u8;
        unsafe { std::ptr::copy_nonoverlapping(src_ptr, buffer.as_mut_ptr(), byte_len) };
        let mut memory_info: *mut OrtMemoryInfo = std::ptr::null_mut();
        let status =
            unsafe { (onnx.create_memory_info)(OrtAllocatorType::Device, OrtMemType::CpuOutput, &mut memory_info as *mut _) };
        if !status.is_null() {
            panic!("Failed to create memory info: {}", onnx.status_to_string(status));
        }
        let shape_i64: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
        let mut value: *mut OrtValue = std::ptr::null_mut();
        let status = unsafe {
            (onnx.create_tensor)(
                memory_info,
                buffer.as_mut_ptr() as *mut c_void,
                byte_len,
                shape_i64.as_ptr(),
                shape_i64.len(),
                T::element_type(),
                &mut value as *mut _,
            )
        };
        if !status.is_null() {
            unsafe { (onnx.release_memory_info)(memory_info) };
            panic!("Failed to create tensor: {}", onnx.status_to_string(status));
        }
        unsafe { (onnx.release_memory_info)(memory_info) };
        Value {
            onnx: Arc::clone(&onnx),
            value,
            _data: buffer,
        }
    }

    pub fn as_slice_mut<T: TensorElement>(&mut self) -> &mut [T] {
        let count = self._data.len() / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts_mut(self._data.as_mut_ptr() as *mut T, count) }
    }

    pub fn empty_typed(onnx: &Arc<Onnx>, shape: &[usize], element_type: ONNXTensorElementDataType) -> Self {
        let buffer = vec![0u8; 0].into_boxed_slice();

        let mut memory_info: *mut OrtMemoryInfo = std::ptr::null_mut();
        let status =
            unsafe { (onnx.create_memory_info)(OrtAllocatorType::Device, OrtMemType::CpuOutput, &mut memory_info as *mut _) };
        if !status.is_null() {
            panic!("Failed to create memory info: {}", onnx.status_to_string(status));
        }
        let shape_i64: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
        let mut value: *mut OrtValue = std::ptr::null_mut();
        let status = unsafe {
            (onnx.create_tensor)(
                memory_info,
                buffer.as_ptr() as *mut c_void,
                0,
                shape_i64.as_ptr(),
                shape_i64.len(),
                element_type,
                &mut value as *mut _,
            )
        };
        if !status.is_null() {
            unsafe { (onnx.release_memory_info)(memory_info) };
            panic!("Failed to create tensor: {}", onnx.status_to_string(status));
        }
        unsafe { (onnx.release_memory_info)(memory_info) };
        Value {
            onnx: Arc::clone(&onnx),
            value,
            _data: buffer,
        }
    }

    pub fn zeros<T: TensorElement + Default>(onnx: &Arc<Onnx>, shape: &[i64]) -> Self {
        let resolved: Vec<usize> = shape.iter().map(|&d| if d < 0 { 1 } else { d as usize }).collect();
        let total: usize = resolved.iter().product();
        let data = vec![T::default(); total];
        Self::from_slice(onnx, &resolved, &data)
    }

    pub fn extract_tensor<T: TensorElement>(&self) -> &[T] {
        let mut type_info: *mut OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        let status = unsafe { (self.onnx.get_tensor_type_and_shape)(self.value, &mut type_info as *mut _) };
        if !status.is_null() {
            panic!("Failed to get tensor type and shape: {}", self.onnx.status_to_string(status));
        }
        let mut element_type = ONNXTensorElementDataType::Undefined;
        let status = unsafe { (self.onnx.get_tensor_element_type)(type_info, &mut element_type as *mut _) };
        if !status.is_null() {
            unsafe { (self.onnx.release_tensor_type_and_shape_info)(type_info) };
            panic!("Failed to get tensor element type: {}", self.onnx.status_to_string(status));
        }
        let mut elem_count: usize = 0;
        let status = unsafe { (self.onnx.get_tensor_shape_element_count)(type_info, &mut elem_count as *mut _) };
        if !status.is_null() {
            unsafe { (self.onnx.release_tensor_type_and_shape_info)(type_info) };
            panic!(
                "Failed to get tensor shape element count: {}",
                self.onnx.status_to_string(status)
            );
        }
        unsafe { (self.onnx.release_tensor_type_and_shape_info)(type_info) };
        if element_type != T::element_type() {
            panic!(
                "Element type mismatch: expected {:?}, got {:?}",
                T::element_type(),
                element_type
            );
        }
        if elem_count == 0 {
            return &[];
        }
        let mut data_ptr: *mut c_void = std::ptr::null_mut();
        let status = unsafe { (self.onnx.get_tensor_mutable_data)(self.value, &mut data_ptr as *mut _) };
        if !status.is_null() {
            panic!("Failed to get tensor mutable data: {}", self.onnx.status_to_string(status));
        }
        unsafe { std::slice::from_raw_parts(data_ptr as *const T, elem_count) }
    }

    pub fn extract_as_f32(&self) -> Vec<f32> {
        let elem_type = self.tensor_element_type();
        match elem_type {
            ONNXTensorElementDataType::Float => {
                let data = self.extract_tensor::<f32>();
                data.to_vec()
            }
            ONNXTensorElementDataType::Float16 => {
                let mut type_info: *mut OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
                let status = unsafe { (self.onnx.get_tensor_type_and_shape)(self.value, &mut type_info as *mut _) };
                if !status.is_null() {
                    panic!("Failed to get tensor type and shape: {}", self.onnx.status_to_string(status));
                }
                let mut elem_count: usize = 0;
                let status = unsafe { (self.onnx.get_tensor_shape_element_count)(type_info, &mut elem_count as *mut _) };
                if !status.is_null() {
                    unsafe { (self.onnx.release_tensor_type_and_shape_info)(type_info) };
                    panic!(
                        "Failed to get tensor shape element count: {}",
                        self.onnx.status_to_string(status)
                    );
                }
                unsafe { (self.onnx.release_tensor_type_and_shape_info)(type_info) };
                if elem_count == 0 {
                    return Vec::new();
                }
                let mut data_ptr: *mut c_void = std::ptr::null_mut();
                let status = unsafe { (self.onnx.get_tensor_mutable_data)(self.value, &mut data_ptr as *mut _) };
                if !status.is_null() {
                    panic!("Failed to get tensor mutable data: {}", self.onnx.status_to_string(status));
                }
                let f16_data = unsafe { std::slice::from_raw_parts(data_ptr as *const u16, elem_count) };
                f16_data.iter().map(|&h| f16_to_f32(h)).collect()
            }
            other => panic!("extract_as_f32: unsupported element type {:?}", other),
        }
    }

    pub fn tensor_shape(&self) -> Vec<i64> {
        let mut type_info: *mut OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        let status = unsafe { (self.onnx.get_tensor_type_and_shape)(self.value, &mut type_info as *mut _) };
        if !status.is_null() {
            panic!("Failed to get tensor type and shape: {}", self.onnx.status_to_string(status));
        }
        let mut dim_count: usize = 0;
        let status = unsafe { (self.onnx.get_dimensions_count)(type_info, &mut dim_count as *mut _) };
        if !status.is_null() {
            unsafe { (self.onnx.release_tensor_type_and_shape_info)(type_info) };
            panic!("Failed to get dimensions count: {}", self.onnx.status_to_string(status));
        }
        let mut dims = vec![0i64; dim_count];
        let status = unsafe { (self.onnx.get_dimensions)(type_info, dims.as_mut_ptr(), dim_count) };
        if !status.is_null() {
            unsafe { (self.onnx.release_tensor_type_and_shape_info)(type_info) };
            panic!("Failed to get dimensions: {}", self.onnx.status_to_string(status));
        }
        unsafe { (self.onnx.release_tensor_type_and_shape_info)(type_info) };
        dims
    }

    pub fn tensor_element_type(&self) -> ONNXTensorElementDataType {
        let mut type_info: *mut OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        let status = unsafe { (self.onnx.get_tensor_type_and_shape)(self.value, &mut type_info as *mut _) };
        if !status.is_null() {
            panic!("Failed to get tensor type and shape: {}", self.onnx.status_to_string(status));
        }
        let mut element_type = ONNXTensorElementDataType::Undefined;
        let status = unsafe { (self.onnx.get_tensor_element_type)(type_info, &mut element_type as *mut _) };
        if !status.is_null() {
            unsafe { (self.onnx.release_tensor_type_and_shape_info)(type_info) };
            panic!("Failed to get tensor element type: {}", self.onnx.status_to_string(status));
        }
        unsafe { (self.onnx.release_tensor_type_and_shape_info)(type_info) };
        element_type
    }

    pub(crate) fn as_ptr(&self) -> *const OrtValue {
        self.value
    }

    pub(crate) unsafe fn from_raw(onnx: &Arc<Onnx>, value: *mut OrtValue) -> Self {
        Value {
            onnx: Arc::clone(&onnx),
            value,
            _data: Box::new([]),
        }
    }

    pub fn deepclone(&self) -> Self {
        let shape = self.tensor_shape();
        let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let elem_type = self.tensor_element_type();
        match elem_type {
            ONNXTensorElementDataType::Float => Self::from_slice(&self.onnx, &shape_usize, self.extract_tensor::<f32>()),
            ONNXTensorElementDataType::Int64 => Self::from_slice(&self.onnx, &shape_usize, self.extract_tensor::<i64>()),
            ONNXTensorElementDataType::Bool => Self::from_slice(&self.onnx, &shape_usize, self.extract_tensor::<bool>()),
            other => panic!("Unsupported element type {:?} for deepclone", other),
        }
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        if !self.value.is_null() {
            unsafe { (self.onnx.release_value)(self.value) };
        }
    }
}
