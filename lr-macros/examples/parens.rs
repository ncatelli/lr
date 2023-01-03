use lr_core::*;
use lr_macros::parser_gen;
use std::hash::Hash;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ParensGrammarTokenKind {
    Epsilon,
    Eof,
    LeftParen,
    RightParen,
}

impl TryFrom<&str> for ParensGrammarTokenKind {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "<epsilon>" => Ok(Self::Epsilon),
            "<eof>" => Ok(Self::Eof),
            "LeftParen" | "(" => Ok(Self::LeftParen),
            "RightParen" | ")" => Ok(Self::RightParen),
            other => Err(format!("unknown variant: {:?}", other)),
        }
    }
}

impl<'a> TerminalVariant<'a> for ParensGrammarTokenKind {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ParensGrammarToken {
    Epsilon,
    Eof,
    LeftParen,
    RightParen,
}

impl<'a> TerminalRepresentable<'a> for ParensGrammarToken {
    type VariantRepr = ParensGrammarTokenKind;

    const EPSILON_VARIANT: Self::VariantRepr = Self::VariantRepr::Epsilon;
    const EOF_VARIANT: Self::VariantRepr = Self::VariantRepr::Eof;

    fn variant(&self) -> Self::VariantRepr {
        match self {
            ParensGrammarToken::Epsilon => Self::EPSILON_VARIANT,
            ParensGrammarToken::Eof => Self::EOF_VARIANT,
            ParensGrammarToken::LeftParen => Self::VariantRepr::LeftParen,
            ParensGrammarToken::RightParen => Self::VariantRepr::RightParen,
        }
    }
}

#[derive(Debug)]
enum ParensGrammarSymbol {
    Goal,
    Parens,
}

fn main() {
    use ParensGrammarTokenKind::*;

    parser_gen!(
        "
        <parens> ::= ( <parens> ) { ParensGrammarSymbol::Parens }
        <parens> ::= ( ) { ParensGrammarSymbol::Parens }
        <parens> ::= <parens> ( ) { ParensGrammarSymbol::Parens }
        <parens> ::= <parens> ( <parens> ) { ParensGrammarSymbol::Parens }
        <parens> ::= ( ) <parens> { ParensGrammarSymbol::Parens }
        <parens> ::= ( <parens> ) <parens> { ParensGrammarSymbol::Parens }
    "
    );
}
