use proc_macro2::TokenStream;
use syn::{parse_macro_input, spanned::Spanned, DeriveInput};

fn parse(input: DeriveInput) -> Result<(), syn::Error> {
    let _input_span = input.span();

    todo!()
}

/// The dispatcher method for tokens annotated with the Lr1 derive.
#[proc_macro_derive(Lr1, attributes(rule))]
pub fn relex(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    parse(input)
        .map(|_| TokenStream::new())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
