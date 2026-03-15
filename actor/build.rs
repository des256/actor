fn main() {
    // ONNX
    println!("cargo:rustc-link-lib=onnxruntime");
    println!("cargo:rustc-link-search=native=/usr/local/lib");

    // CUDA runtime
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/targets/aarch64-linux/lib");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/targets/x86_64-linux/lib");
    println!("cargo:rustc-link-lib=dylib=cudart");

    // TensorRT
    println!("cargo:rustc-link-search=native=/usr/local/tensorrt/lib");
    println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-lib=dylib=nvinfer");

    // build TensorRT C++ to C API bindings
    cc::Build::new()
        .cpp(true)
        .include("/usr/local/cuda-12.6/include")
        .std("c++17")
        .file("src/tensorrt/ffi.cpp")
        .compile("trt_runtime_stub");
}
