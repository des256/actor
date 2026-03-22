use std::ffi::CStr;

pub enum DataType {
    Bool,
    Int8,
    Int32,
    Int64,
    Float32,
    Float16,
}

impl DataType {
    fn from_ffi(value: i32) -> Self {
        match value {
            0 => DataType::Float32,
            1 => DataType::Float16,
            2 => DataType::Int8,
            3 => DataType::Int32,
            4 => DataType::Bool,
            5 => DataType::Int64,
            _ => DataType::Float32,
        }
    }

    pub fn byte_size(&self) -> usize {
        match self {
            DataType::Float32 | DataType::Int32 => 4,
            DataType::Float16 => 2,
            DataType::Int8 | DataType::Bool => 1,
            DataType::Int64 => 8,
        }
    }
}

pub struct TensorInfo {
    pub name: String,
    pub is_input: bool,
    pub dtype: DataType,
    pub shape: Vec<i64>,
}

pub(crate) fn last_error() -> String {
    unsafe {
        let ptr = ffi::trt_get_last_error();
        if ptr.is_null() {
            "Unknown TensorRT error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

pub mod ffi;

mod buffer;
pub use buffer::*;

mod tensorrt;
pub use tensorrt::*;

mod engine;
pub use engine::*;

mod context;
pub use context::*;
