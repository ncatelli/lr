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
