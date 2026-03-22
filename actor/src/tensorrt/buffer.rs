use {
    super::*,
    std::{ffi::c_void, ptr::null_mut},
};

const CUDA_MEMCPY_H2D: i32 = 1;
const CUDA_MEMCPY_D2H: i32 = 2;
const CUDA_MEMCPY_D2D: i32 = 3;

pub struct Buffer {
    pub ptr: *mut c_void,
    size: usize,
}

impl Buffer {
    pub fn new(bytes: usize) -> Self {
        let mut ptr: *mut c_void = null_mut();
        let rc = unsafe { ffi::cudaMalloc(&mut ptr, bytes) };
        assert!(rc == 0, "cudaMalloc({}) failed", bytes);
        unsafe { ffi::cudaMemset(ptr, 0, bytes) };
        Self { ptr, size: bytes }
    }

    pub fn upload<T>(&self, data: &[T]) {
        let bytes = data.len() * size_of::<T>();
        assert!(bytes <= self.size, "upload overflow: {} > {}", bytes, self.size);
        unsafe { ffi::cudaMemcpy(self.ptr, data.as_ptr() as *const c_void, bytes, CUDA_MEMCPY_H2D) };
    }

    pub fn download<T>(&self, data: &mut [T]) {
        let bytes = data.len() * size_of::<T>();
        assert!(bytes <= self.size, "download overflow: {} > {}", bytes, self.size);
        unsafe { ffi::cudaMemcpy(data.as_mut_ptr() as *mut c_void, self.ptr, bytes, CUDA_MEMCPY_D2H) };
    }

    pub fn upload_async<T>(&self, data: &[T], stream: *mut c_void) {
        let bytes = data.len() * size_of::<T>();
        assert!(bytes <= self.size, "upload_async overflow: {} > {}", bytes, self.size);
        unsafe { ffi::cudaMemcpyAsync(self.ptr, data.as_ptr() as *const c_void, bytes, CUDA_MEMCPY_H2D, stream) };
    }

    pub fn copy_from_device(&self, src: &Buffer, bytes: usize) {
        assert!(bytes <= self.size && bytes <= src.size, "D2D copy overflow");
        unsafe { ffi::cudaMemcpy(self.ptr, src.ptr, bytes, CUDA_MEMCPY_D2D) };
    }

    pub fn copy_from_device_async(&self, src_ptr: *mut c_void, bytes: usize, stream: *mut c_void) {
        assert!(bytes <= self.size, "D2D async copy overflow");
        unsafe { ffi::cudaMemcpyAsync(self.ptr, src_ptr as *const c_void, bytes, CUDA_MEMCPY_D2D, stream) };
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { ffi::cudaFree(self.ptr) };
    }
}
