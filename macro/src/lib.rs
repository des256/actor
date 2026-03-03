use proc_macro::TokenStream;

const _DART_PATH: &str = "webui/lib/rstypes";

#[proc_macro_derive(Codec)]
pub fn derive_codec(_input: TokenStream) -> TokenStream {
    // parse struct, tuple or enum
    TokenStream::new()
}

#[proc_macro_derive(Dart)]
pub fn derive_dart(_input: TokenStream) -> TokenStream {
    TokenStream::new()
}
