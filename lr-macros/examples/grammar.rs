use lr_core::*;
use lr_macros::parser_gen;
use std::hash::Hash;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum GrammarTokenKind {
    Epsilon,
    Eof,
    Minus,
    Star,
    One,
}

impl TryFrom<&str> for GrammarTokenKind {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "<epsilon>" => Ok(Self::Epsilon),
            "<eof>" => Ok(Self::Eof),
            "Minus" | "-" => Ok(Self::Minus),
            "Star" | "*" => Ok(Self::Star),
            "1" => Ok(Self::One),
            other => Err(format!("unknown variant: {:?}", other)),
        }
    }
}

impl<'a> TerminalVariant<'a> for GrammarTokenKind {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum GrammarToken {
    Epsilon,
    Eof,
    Minus,
    Star,
    One,
}

impl<'a> TerminalRepresentable<'a> for GrammarToken {
    type VariantRepr = GrammarTokenKind;

    const EPSILON_VARIANT: Self::VariantRepr = Self::VariantRepr::Epsilon;
    const EOF_VARIANT: Self::VariantRepr = Self::VariantRepr::Eof;

    fn variant(&self) -> Self::VariantRepr {
        match self {
            GrammarToken::Epsilon => Self::EPSILON_VARIANT,
            GrammarToken::Eof => Self::EOF_VARIANT,
            GrammarToken::Minus => Self::VariantRepr::Minus,
            GrammarToken::Star => Self::VariantRepr::Star,
            GrammarToken::One => Self::VariantRepr::One,
        }
    }
}

#[derive(Debug)]
enum GrammarSymbol {
    Goal,
    Expr,
    Factor,
    Term,
}

fn main() {
    use GrammarTokenKind::*;

    parser_gen!(
        "
        %type GrammarSymbol
        %token GrammarTokenKind
        <E> ::= <T> - <E> { GrammarSymbol::Expr }
        <E> ::= <T> { GrammarSymbol::Expr }
        <T> ::= <F> * <T> { GrammarSymbol::Term }
        <T> ::= <F> { GrammarSymbol::Term }
        <F> ::= 1 { GrammarSymbol::Factor }
    "
    );
}
