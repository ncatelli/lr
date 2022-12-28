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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ParensGrammarToken {
    Epsilon,
    Eof,
    LeftParen,
    RightParen,
}

impl TerminalRepresentable for ParensGrammarToken {
    type VariantRepr = ParensGrammarTokenKind;

    const EPSILON_VARIANT: Self::VariantRepr = Self::VariantRepr::Epsilon;
    const EOF_VARIANT: Self::VariantRepr = Self::VariantRepr::Eof;

    fn variant(&self) -> Self::VariantRepr {
        match self {
            ParensGrammarToken::Epsilon => Self::VariantRepr::Epsilon,
            ParensGrammarToken::Eof => Self::VariantRepr::Eof,
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

impl NonTerminalRepresentable for ParensGrammarSymbol {
    type VariantRepr = ParensGrammarSymbolKind;

    const GOAL_VARIANT: Self::VariantRepr = Self::VariantRepr::Goal;

    fn variant(&self) -> Self::VariantRepr {
        match self {
            ParensGrammarSymbol::Goal => Self::VariantRepr::Goal,
            ParensGrammarSymbol::Parens => Self::VariantRepr::Parens,
        }
    }
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
