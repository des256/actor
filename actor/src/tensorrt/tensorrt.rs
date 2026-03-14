use std::{ptr::null_mut, sync::Arc};

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

fn last_error() -> String {
    unsafe {
        let ptr = ffi::trt_get_last_error();
        if ptr.is_null() {
            "Unknown TensorRT error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

pub struct Tensorrt {
    runtime: *mut ffi::TrtRuntime,
}

unsafe impl Send for Tensorrt {}

pub struct Engine {
    tensorrt: Arc<Tensorrt>,
    engine: *mut ffi::TrtEngine,
}

unsafe impl Send for Engine {}

impl Tensorrt {
    pub fn new() -> Arc<Self> {
        let mut runtime = null_mut();
        if unsafe { ffi::trt_runtime_create(&mut runtime) } != ffi::TrtStatus::Ok {
            panic!("failed to create TensorRT runtime: {}", last_error());
        }
        Arc::new(Self { runtime })
    }

    pub fn load_engine(&mut self, path: &str) -> Arc<Engine> {
        let c_path = match CString::new(path) {
            Ok(c_path) => c_path,
            Err(error) => panic!("Null byte in engine path: {}", error),
        };
        let mut engine = null_mut();
        if unsafe { ffi::trt_engine_load(self.runtime, c_path.as_ptr(), &mut engine) } != ffi::TrtStatus::Ok {
            panic!("failed to load engine from {}: {}", path, last_error());
        }
        Arc::new(Engine {
            tensorrt: Arc::clone(&self),
            engine,
        })
    }
}

impl Drop for Tensorrt {
    fn drop(&mut self) {
        if !self.runtime.is_null() {
            unsafe { ffi::trt_runtime_destroy(self.runtime) };
        }
    }
}

pub struct Context {
    engine: Arc<Engine>,
    context: *mut ffi::TrtContext,
}

unsafe impl Send for Context {}

impl Engine {
    pub fn get_io_tensors(&self) -> Vec<TensorInfo> {
        let n = unsafe { ffi::trt_engine_get_num_io_tensors(self.engine) };
        let mut result = Vec::with_capacity(n as usize);

        for i in 0..n {
            let name_ptr = unsafe { ffi::trt_engine_get_io_tensor_name(self.engine, i) };
            if name_ptr.is_null() {
                continue;
            }
            let name = unsafe { CStr::from_ptr(name_ptr) }.to_string_lossy().into_owned();
            let c_name = CString::new(name.as_str()).unwrap();

            let io_mode = unsafe { ffi::trt_engine_get_tensor_io_mode(self.engine, c_name.as_ptr()) };
            let dtype_raw = unsafe { ffi::trt_engine_get_tensor_dtype(self.engine, c_name.as_ptr()) };

            let mut dims = [0i64; 16];
            let ndims = unsafe { ffi::trt_engine_get_tensor_shape(self.engine, c_name.as_ptr(), dims.as_mut_ptr(), 16) };

            result.push(TensorInfo {
                name,
                is_input: io_mode == 0,
                dtype: DataType::from_ffi(dtype_raw),
                shape: dims[..ndims as usize].to_vec(),
            });
        }   result
    }

    pub fn create_context(&self) -> Arc<Self> {
        let mut context = null_mut();
        if let 
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if !self.engine.is_null() {
            unsafe { ffi::trt_engine_destroy(self.engine) };
        }
    }
}