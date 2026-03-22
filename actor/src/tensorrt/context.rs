use {
    super::*,
    std::{
        ffi::{CStr, CString, c_void},
        sync::Arc,
    },
};

pub struct Context {
    pub(crate) engine: Arc<Engine>,
    pub(crate) context: *mut ffi::TrtContext,
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    pub fn set_input_shape(&self, name: &str, dims: &[i64]) {
        let c_name = CString::new(name).unwrap();
        self.set_input_shape_cstr(&c_name, dims);
    }

    pub fn set_input_shape_cstr(&self, name: &CStr, dims: &[i64]) {
        let c_dims = dims.as_ptr();
        let ndims = dims.len() as i32;
        if unsafe { ffi::trt_context_set_input_shape(self.context, name.as_ptr(), c_dims, ndims) } != ffi::TrtStatus::Ok {
            panic!("failed to set input shape: {}", last_error());
        }
    }

    pub fn set_tensor_address(&self, name: &str, ptr: *mut c_void) {
        let c_name = CString::new(name).unwrap();
        self.set_tensor_address_cstr(&c_name, ptr);
    }

    pub fn set_tensor_address_cstr(&self, name: &CStr, ptr: *mut c_void) {
        if unsafe { ffi::trt_context_set_tensor_address(self.context, name.as_ptr(), ptr) } != ffi::TrtStatus::Ok {
            panic!("failed to set tensor address: {}", last_error());
        }
    }

    pub fn enqueue(&self, stream: *mut c_void) {
        if unsafe { ffi::trt_context_enqueue(self.context, stream) } != ffi::TrtStatus::Ok {
            panic!("failed to enqueue: {}", last_error());
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.context.is_null() {
            unsafe { ffi::trt_context_destroy(self.context) };
        }
    }
}
