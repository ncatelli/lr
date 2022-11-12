use crate::grammar::GrammarTable;

#[derive(Debug, PartialEq, Eq)]
pub enum GrammarLoadErrorKind {
    InvalidRule,
    ConflictingRule,
    Other,
}

impl std::fmt::Display for GrammarLoadErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRule => write!(f, "provided rule is invalid",),
            Self::ConflictingRule => write!(f, "provided rule conflicts with existing rule",),
            Self::Other => write!(f, "undefined load error"),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ParserGenError {
    kind: GrammarLoadErrorKind,
    data: Option<String>,
}

impl ParserGenError {
    pub fn new(kind: GrammarLoadErrorKind) -> Self {
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

pub(crate) struct LrParser {}

impl std::fmt::Display for ParserGenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.data {
            Some(ctx) => write!(f, "{}: {}", &self.kind, ctx),
            None => write!(f, "{}", &self.kind),
        }
    }
}

/// Build a LR(1) parser from a given grammar.
pub(crate) fn build_lr_parser(grammar_table: GrammarTable) -> Result<LrParser, ParserGenError> {
    todo!()
}
