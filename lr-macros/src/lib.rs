use lr::LrTable;
use proc_macro2::{Span, TokenStream};
use quote::ToTokens;
use syn::{
    custom_punctuation, parse::Parse, parse_macro_input, spanned::Spanned, Block, Ident, LitStr,
    Token,
};

use grammar::GrammarLoadError;

mod grammar;
mod lr;

/// Represents the kind of table that can be generated
enum GeneratorKind {
    /// LR(1) Grammar
    Lr1,
}

#[derive(Debug, PartialEq, Eq)]
enum ErrorKind {
    GrammarError(GrammarLoadError),
    TableGenerationError(lr::TableGenError),
}

impl std::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GrammarError(err) => write!(f, "grammar error: {}", err),
            Self::TableGenerationError(err) => write!(f, "table generation error: {}", err),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Error {
    kind: ErrorKind,
    data: Option<String>,
}

impl Error {
    pub(crate) fn new(kind: ErrorKind) -> Self {
        Self { kind, data: None }
    }

    pub(crate) fn with_data_mut(&mut self, data: String) {
        self.data = Some(data)
    }

    pub(crate) fn with_data(mut self, data: String) -> Self {
        self.with_data_mut(data);
        self
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.data {
            Some(ctx) => write!(f, "{}: {}", &self.kind, ctx),
            None => write!(f, "{}", &self.kind),
        }
    }
}

fn parse(input: TokenStream) -> syn::Result<LrTable> {
    use grammar::load_grammar;
    use lr::LrTableGenerator;
    let span = input.span();

    let grammar_str = syn::parse2::<LitStr>(input)?.value();
    let grammar_table = load_grammar(&grammar_str)
        .map_err(|e| Error::new(ErrorKind::GrammarError(e)).to_string())
        .map_err(|message| syn::Error::new(span, message))?;

    lr::Lr1::generate_table(&grammar_table)
        .map_err(|e| Error::new(ErrorKind::TableGenerationError(e)).to_string())
        .map_err(|message| syn::Error::new(span, message))
}

fn codegen(_table: LrTable) -> syn::Result<TokenStream> {
    todo!()
}

#[proc_macro]
pub fn parser_gen(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input);

    let _lr_table = parse(input).unwrap();

    proc_macro::TokenStream::new()
}
