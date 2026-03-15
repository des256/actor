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

    // TensorRT-LLM: discover library path from installed Python package
    if let Some(output) = std::process::Command::new("python3")
        .args(["-c", "import tensorrt_llm,os;print(os.path.join(os.path.dirname(tensorrt_llm.__file__),'libs'))"])
        .output()
        .ok()
        .filter(|o| o.status.success())
    {
        let lib_dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
        println!("cargo:rustc-link-search=native={}", lib_dir);
    }
    println!("cargo:rustc-link-lib=dylib=tensorrt_llm");
    println!("cargo:rustc-link-lib=dylib=nvinfer_plugin_tensorrt_llm");

    // build TensorRT C++ to C API bindings
    cc::Build::new()
        .cpp(true)
        .include("/usr/local/cuda-12.6/include")
        .include("/TensorRT-LLM/cpp/include")
        .std("c++17")
        .file("src/tensorrt/ffi.cpp")
        .compile("trt_runtime_stub");
}
