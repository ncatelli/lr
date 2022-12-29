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
            "(" => Ok(Self::LeftParen),
            ")" => Ok(Self::RightParen),
            other => Err(format!("unknown variant: {:?}", other)),
        }
    }
}

impl<'a> grammar::TerminalVariant<'a> for ParensGrammarTokenKind {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ParensGrammarToken {
    Epsilon,
    Eof,
    LeftParen,
    RightParen,
}

impl<'a> grammar::TerminalRepresentable<'a> for ParensGrammarToken {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ParensGrammarSymbolKind {
    Goal,
    Parens,
}

#[derive(Debug)]
enum ParensGrammarSymbol {
    Goal,
    Parens,
}

fn main() {
    use ParensGrammarSymbolKind::*;
    use ParensGrammarTokenKind::*;

    parser_gen! (
        Parens ::= LeftParen Parens RightParen { ParensGrammarSymbol::Parens }
        Parens ::= LeftParen RightParen { ParensGrammarSymbol::Parens }
        Parens ::= Parens LeftParen RightParen { ParensGrammarSymbol::Parens }
        Parens ::= Parens LeftParen Parens RightParen { ParensGrammarSymbol::Parens }
        Parens ::= LeftParen RightParen Parens { ParensGrammarSymbol::Parens }
        Parens ::= LeftParen Parens RightParen Parens { ParensGrammarSymbol::Parens }
    );
}
