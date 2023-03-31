use grammar::GrammarLoadError;

pub mod grammar;
pub mod lr;

/// Represents the kind of table that can be generated
pub enum GeneratorKind {
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

    pub fn with_data_mut(&mut self, data: String) {
        self.data = Some(data)
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalOrNonTerminal<T, NT> {
    Terminal(T),
    NonTerminal(NT),
}

#[allow(unused)]
pub fn generate_table_from_ruleset<G: AsRef<str>>(
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
