use core::hash::Hash;
use std::collections::HashMap;

use grammar::GrammarLoadError;

pub mod grammar;
pub mod lr;

/// Represents the kind of table that can be generated
#[allow(unused)]
enum GeneratorKind {
    /// LR(1) Grammar
    Lr1,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ErrorKind {
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
pub struct Error {
    kind: ErrorKind,
    data: Option<String>,
}

impl Error {
    pub fn new(kind: ErrorKind) -> Self {
        Self { kind, data: None }
    }

    #[allow(unused)]
    pub fn with_data_mut(&mut self, data: String) {
        self.data = Some(data)
    }

    #[allow(unused)]
    pub fn with_data(mut self, data: String) -> Self {
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

#[derive(Debug)]
struct TableGenerationOptions<'a> {
    opts: HashMap<&'a str, &'a str>,
}

impl<'a> TableGenerationOptions<'a> {
    fn new(opts: HashMap<&'a str, &'a str>) -> Self {
        Self { opts }
    }
}

#[derive(Debug)]
#[allow(unused)]
pub struct TableGenerationMetadata {
    pub grammar_table: grammar::GrammarTable,
    pub lr_table: lr::LrTable,
    pub ast_ident: String,
    pub token_ident: String,
}

impl TableGenerationMetadata {
    pub fn new<A: AsRef<str>, T: AsRef<str>>(
        grammar_table: grammar::GrammarTable,
        lr_table: lr::LrTable,
        ast_ident: A,
        token_ident: T,
    ) -> Self {
        let ast_ident = ast_ident.as_ref().to_string();
        let token_ident = token_ident.as_ref().to_string();

        Self {
            grammar_table,
            ast_ident,
            token_ident,
            lr_table,
        }
    }
}
pub trait TerminalVariant<'a>: Copy + Eq + Hash + Ord + TryFrom<&'a str> {
    fn from_str_repr(src: &'a str) -> Option<Self> {
        Self::try_from(src).ok()
    }
}

/// A trait signifying that a type can be represented as a Terminal within the
/// grammar.
pub trait TerminalRepresentable<'a>
where
    Self: Sized,
    Self::VariantRepr: TerminalVariant<'a>,
{
    type VariantRepr;

    const EPSILON_VARIANT: Self::VariantRepr;
    const EOF_VARIANT: Self::VariantRepr;

    fn variant(&self) -> Self::VariantRepr;
}
