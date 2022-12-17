use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(lrSymbol, attributes(terminal, nonterminal))]
pub fn symbol_token_gen(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let _input = parse_macro_input!(input as DeriveInput);
    todo!()
}
