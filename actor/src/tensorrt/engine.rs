use {
    super::*,
    std::{
        ffi::{CStr, CString},
        ptr::null_mut,
        sync::Arc,
    },
};

pub struct Engine {
    pub(crate) tensorrt: Arc<Tensorrt>,
    pub(crate) engine: *mut ffi::TrtEngine,
}

unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

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
        }
        result
    }

    pub fn create_context(self: &Arc<Self>) -> Arc<Context> {
        let mut context = null_mut();
        if unsafe { ffi::trt_context_create(self.engine, &mut context) } != ffi::TrtStatus::Ok {
            panic!("failed to create context: {}", last_error());
        }
        Arc::new(Context {
            engine: Arc::clone(&self),
            context,
        })
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if !self.engine.is_null() {
            unsafe { ffi::trt_engine_destroy(self.engine) };
        }
    }
}
