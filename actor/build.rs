fn main() {
    println!("cargo:rustc-link-lib=onnxruntime");
    println!("cargo:rustc-link-search=native=/usr/local/lib");
}
