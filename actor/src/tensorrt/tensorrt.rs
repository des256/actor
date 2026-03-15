use {
    super::*,
    std::{
        ffi::{CStr, CString, c_void},
        ptr::null_mut,
        sync::Arc,
    },
};

pub struct Tensorrt {
    pub(crate) runtime: *mut ffi::TrtRuntime,
}

unsafe impl Send for Tensorrt {}
unsafe impl Sync for Tensorrt {}

impl Tensorrt {
    pub fn new() -> Arc<Self> {
        let mut runtime = null_mut();
        if unsafe { ffi::trt_runtime_create(&mut runtime) } != ffi::TrtStatus::Ok {
            panic!("failed to create TensorRT runtime: {}", last_error());
        }
        Arc::new(Self { runtime })
    }

    pub fn load_engine(self: &Arc<Self>, path: &str) -> Arc<Engine> {
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
