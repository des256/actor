use std::{ffi::c_void, ptr::null_mut};

unsafe extern "C" {
    pub fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    pub fn cudaFree(ptr: *mut c_void) -> i32;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    pub fn cudaMemset(ptr: *mut c_void, value: i32, count: usize) -> i32;
    pub fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
    pub fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    pub fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
}

const CUDA_MEMCPY_H2D: i32 = 1;
const CUDA_MEMCPY_D2H: i32 = 2;

pub struct Buffer {
    pub ptr: *mut c_void,
    size: usize,
}

impl Buffer {
    pub fn new(bytes: usize) -> Self {
        let mut ptr: *mut c_void = null_mut();
        let rc = unsafe { cudaMalloc(&mut ptr, bytes) };
        assert!(rc == 0, "cudaMalloc({}) failed", bytes);
        unsafe { cudaMemset(ptr, 0, bytes) };
        Self { ptr, size: bytes }
    }

    pub fn upload<T>(&self, data: &[T]) {
        let bytes = data.len() * size_of::<T>();
        assert!(bytes <= self.size, "upload overflow: {} > {}", bytes, self.size);
        unsafe { cudaMemcpy(self.ptr, data.as_ptr() as *const c_void, bytes, CUDA_MEMCPY_H2D) };
    }

    pub fn download<T>(&self, data: &mut [T]) {
        let bytes = data.len() * size_of::<T>();
        assert!(bytes <= self.size, "download overflow: {} > {}", bytes, self.size);
        unsafe { cudaMemcpy(data.as_mut_ptr() as *mut c_void, self.ptr, bytes, CUDA_MEMCPY_D2H) };
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { cudaFree(self.ptr) };
    }
}
