pub mod grammar;
pub mod lr;

/// Represents the kind of table that can be generated
pub enum GeneratorKind {
    /// LR(1) Grammar
    Lr1,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ErrorKind {
    GrammarError(grammar::GrammarLoadError),
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

/// A top-level error type for wrapping both grammar parsing and table
/// generation errors.
#[derive(Debug, PartialEq, Eq)]
pub struct Error {
    kind: ErrorKind,
    data: Option<String>,
}

impl Error {
    pub fn new(kind: ErrorKind) -> Self {
        Self { kind, data: None }
    }

    /// Allows a caller to enrich an error with a string signifying additional
    /// information about the error.
    pub fn with_data_mut<S: AsRef<str>>(&mut self, data: S) {
        let data = data.as_ref().to_string();

        self.data = Some(data)
    }

    /// Allows a caller to enrich an error with a string signifying additional
    /// information about the error.
    pub fn with_data<S: AsRef<str>>(mut self, data: S) -> Self {
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

/// A trait for flagging that a type can be represented as a nonterminal in a
/// generated parser.
pub trait NonTerminalRepresentable: std::fmt::Debug
where
    Self::Terminal: TerminalRepresentable,
{
    // A Representation of a Terminal
    type Terminal;
}

/// A trait for flagging that a type can be represented as a terminal in a
/// generated parser.
pub trait TerminalRepresentable: std::fmt::Debug
where
    Self::Repr: std::fmt::Debug + Copy + Eq,
{
    /// A copyable representation for uses in internal matching.
    type Repr;

    fn to_variant_repr(&self) -> Self::Repr;

    /// The end of file terminal
    fn eof() -> Self::Repr;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalOrNonTerminal<T, NT> {
    Terminal(T),
    NonTerminal(NT),
}

#[allow(unused)]
pub fn generate_table_from_production_set<G: AsRef<str>>(
    kind: GeneratorKind,
    grammar: G,
) -> Result<lr::LrTable, Error> {
    use grammar::load_grammar;

    let grammar = grammar.as_ref();
    let grammar_table =
        load_grammar(grammar).map_err(|e| Error::new(ErrorKind::GrammarError(e)))?;

    match kind {
        GeneratorKind::Lr1 => {
            use crate::lr::LrTableGenerator;

            crate::lr::Lr1::generate_table(&grammar_table)
                .map_err(|e| Error::new(ErrorKind::TableGenerationError(e)))
        }
    }
}

#[allow(unused)]
pub fn generate_table_from_grammar(
    kind: GeneratorKind,
    grammar_table: &grammar::GrammarTable,
) -> Result<lr::LrTable, Error> {
    match kind {
        GeneratorKind::Lr1 => {
            use crate::lr::LrTableGenerator;

            crate::lr::Lr1::generate_table(grammar_table)
                .map_err(|e| Error::new(ErrorKind::TableGenerationError(e)))
        }
    }
}
