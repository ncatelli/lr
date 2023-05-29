use lr_core::prelude::v1::*;
use lr_core::TerminalOrNonTerminal;
pub use lr_derive::Lr1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalKind {
    Plus,
    #[allow(unused)]
    Star,
    Int,
    Eof,
}

impl std::fmt::Display for TerminalKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Plus => write!(f, "+"),
            Self::Star => write!(f, "*"),
            Self::Int => write!(f, "int",),
            Self::Eof => write!(f, "<$>"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Terminal<'a> {
    variant: TerminalKind,
    data: &'a str,
}

impl<'a> Terminal<'a> {
    pub fn new(variant: TerminalKind, data: &'a str) -> Self {
        Self { variant, data }
    }
}

impl<'a> std::fmt::Display for Terminal<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.variant {
            TerminalKind::Plus | TerminalKind::Star | TerminalKind::Eof => {
                write!(f, "{}", self.variant)
            }
            TerminalKind::Int => write!(f, "{}", self.data),
        }
    }
}

impl<'a> TerminalRepresentable for Terminal<'a> {
    type Repr = TerminalKind;

    fn to_variant_repr(&self) -> Self::Repr {
        self.variant
    }

    fn eof() -> Self::Repr {
        Self::Repr::Eof
    }
}

#[derive(Debug, PartialEq)]
pub enum NonTermKind<'a> {
    Mul(NonTerminal<'a>, NonTerminal<'a>),
    Add(NonTerminal<'a>, NonTerminal<'a>),
    Unary(NonTerminal<'a>),
}

type TermOrNonTerm<'a> = TerminalOrNonTerminal<Terminal<'a>, NonTerminal<'a>>;

#[allow(unused)]
fn reduce_b_non_term<'a>(elems: &mut Vec<TermOrNonTerm<'a>>) -> Result<NonTerminal<'a>, String> {
    if let Some(TermOrNonTerm::Terminal(term)) = elems.pop() {
        Ok(NonTerminal::B(term))
    } else {
        Err("expected terminal at top of stack in reducer.".to_string())
    }
}

#[allow(unused)]
fn reduce_e_unary_non_term<'a>(
    elems: &mut Vec<TermOrNonTerm<'a>>,
) -> Result<NonTerminal<'a>, String> {
    if let Some(TermOrNonTerm::NonTerminal(nonterm)) = elems.pop() {
        let non_term_kind = NonTermKind::Unary(nonterm);

        Ok(NonTerminal::E(Box::new(non_term_kind)))
    } else {
        Err("expected non-terminal at top of stack in production 3 reducer.".to_string())
    }
}

#[allow(unused)]
fn reduce_e_binary_non_term<'a>(
    production_id: usize,
    elems: &mut Vec<TermOrNonTerm<'a>>,
) -> Result<NonTerminal<'a>, String> {
    let optional_rhs = elems.pop();
    let optional_term = elems.pop();
    let optional_lhs = elems.pop();
    let err_msg = format!(
        "expected 3 elements at top of stack in production {} reducer. got [{:?}, {:?}, {:?}]",
        production_id, &optional_lhs, &optional_term, &optional_rhs
    );

    // reversed due to popping elements
    if let [Some(TermOrNonTerm::NonTerminal(lhs)), Some(TermOrNonTerm::Terminal(op)), Some(TerminalOrNonTerminal::NonTerminal(rhs))] =
        [optional_lhs, optional_term, optional_rhs]
    {
        let non_term_kind = match op.variant {
            TerminalKind::Star => NonTermKind::Mul(lhs, rhs),
            TerminalKind::Plus => NonTermKind::Add(lhs, rhs),
            // Dispatcher should never reach this block of code due to parser guarantees.
            _ => unreachable!(),
        };

        Ok(NonTerminal::E(Box::new(non_term_kind)))
    } else {
        Err(err_msg)
    }
}

#[allow(unused)]
fn reduce_goal<'a>(elems: &mut Vec<TermOrNonTerm<'a>>) -> Result<NonTerminal<'a>, String> {
    if let Some(TermOrNonTerm::NonTerminal(NonTerminal::E(e))) = elems.pop() {
        Ok(NonTerminal::E(e))
    } else {
        Err("expected non-terminal at top of stack in production 3 reducer.".to_string())
    }
}

#[derive(Debug, Lr1, PartialEq)]
pub enum NonTerminal<'a> {
    #[goal(r"<E>", reduce_goal)]
    #[production(r"<E> TerminalKind::Star <B>", |elems| { reduce_e_binary_non_term(2, elems) })]
    #[production(r"<E> TerminalKind::Plus <B>", |elems| { reduce_e_binary_non_term(3, elems) })]
    #[production(r"<B>", reduce_e_unary_non_term)]
    E(Box<NonTermKind<'a>>),
    #[production(r"TerminalKind::Int", reduce_b_non_term)]
    B(Terminal<'a>),
}

impl<'a> NonTerminalRepresentable for NonTerminal<'a> {
    type Terminal = Terminal<'a>;
}

#[test]
fn derived_macro_generator_should_parse_tokens_with_embedded_values() {
    let data = "10 + 1";
    let input = [
        Terminal::new(TerminalKind::Int, &data[0..2]),
        Terminal::new(TerminalKind::Plus, &data[3..4]),
        Terminal::new(TerminalKind::Int, &data[5..6]),
        Terminal::new(TerminalKind::Eof, &data[6..]),
    ];
    let tokenizer = input.into_iter();

    let parse_tree = NonTerminal::parse_input(tokenizer);

    let expected = NonTerminal::E(Box::new(NonTermKind::Add(
        NonTerminal::E(Box::new(NonTermKind::Unary(NonTerminal::B(Terminal::new(
            TerminalKind::Int,
            &data[0..2],
        ))))),
        NonTerminal::B(Terminal::new(TerminalKind::Int, &data[5..6])),
    )));

    assert_eq!(parse_tree, Ok(expected));
}
