use lr_core::TerminalOrNonTerminal;
pub use lr_derive::Lr1;
pub use relex_derive::Relex;

#[derive(Relex, Debug, PartialEq, Eq)]
pub enum Terminal {
    #[matches(r"+")]
    Plus,
    #[matches(r"*")]
    Star,
    #[matches(r"0")]
    Zero,
    #[matches(r"1")]
    One,
}

#[derive(Debug, PartialEq)]
pub enum NonTermKind {
    Mul(NonTerminal, NonTerminal),
    Add(NonTerminal, NonTerminal),
    Unary(NonTerminal),
}

type TermOrNonTerm = TerminalOrNonTerminal<Terminal, NonTerminal>;

#[allow(unused)]
fn reduce_b_non_term(elems: &mut Vec<TermOrNonTerm>) -> Result<NonTerminal, String> {
    if let Some(TermOrNonTerm::Terminal(term)) = elems.pop() {
        Ok(NonTerminal::B(term))
    } else {
        Err("expected terminal at top of stack in reducer.".to_string())
    }
}

#[derive(Debug, Lr1, PartialEq)]
pub enum NonTerminal {
    #[goal(r"<E>")]
    #[rule(r"<E> Terminal::Star <B>")]
    #[rule(r"<E> Terminal::Plus <B>")]
    #[rule(r"<B>")]
    E(Box<NonTermKind>),
    #[rule(r"Terminal::Zero", reduce_b_non_term)]
    #[rule(r"Terminal::One", reduce_b_non_term)]
    B(Terminal),
}

fn main() {}
