use lr_core::{
    LrParseable, NonTerminalRepresentable, TerminalOrNonTerminal, TerminalRepresentable,
};
pub use lr_derive::Lr1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalKind {
    Plus,
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
pub enum Terminal {
    Plus,
    #[allow(unused)]
    Star,
    Int(i64),
    Eof,
}

impl std::fmt::Display for Terminal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Plus => write!(f, "+"),
            Self::Star => write!(f, "*"),
            Self::Int(i) => write!(f, "{}", i),
            Self::Eof => write!(f, "<$>"),
        }
    }
}

impl TerminalRepresentable for Terminal {
    /// the associated type representing the variant kind.
    type Repr = TerminalKind;

    fn eof() -> Self::Repr {
        Self::Repr::Eof
    }

    fn to_variant_repr(&self) -> Self::Repr {
        match self {
            Self::Plus => Self::Repr::Plus,
            Self::Star => Self::Repr::Star,
            Self::Int(_) => Self::Repr::Int,
            Self::Eof => Self::Repr::Eof,
        }
    }
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

#[allow(unused)]
fn reduce_e_unary_non_term(elems: &mut Vec<TermOrNonTerm>) -> Result<NonTerminal, String> {
    if let Some(TermOrNonTerm::NonTerminal(nonterm)) = elems.pop() {
        let non_term_kind = NonTermKind::Unary(nonterm);

        Ok(NonTerminal::E(Box::new(non_term_kind)))
    } else {
        Err(format!(
            "expected non-terminal at top of stack in production 3 reducer.",
        ))
    }
}

#[allow(unused)]
fn reduce_e_binary_non_term(
    production_id: usize,
    elems: &mut Vec<TermOrNonTerm>,
) -> Result<NonTerminal, String> {
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
        let non_term_kind = match op {
            Terminal::Star => NonTermKind::Mul(lhs, rhs),
            Terminal::Plus => NonTermKind::Add(lhs, rhs),
            // Dispatcher should never reach this block of code due to parser guarantees.
            _ => unreachable!(),
        };

        Ok(NonTerminal::E(Box::new(non_term_kind)))
    } else {
        Err(err_msg)
    }
}

#[derive(Debug, Lr1, PartialEq)]
pub enum NonTerminal {
    #[goal(r"<E>", reduce_e_unary_non_term)]
    #[production(r"<E> TerminalKind::Star <B>", |elems| { reduce_e_binary_non_term(2, elems) })]
    #[production(r"<E> TerminalKind::Plus <B>", |elems| { reduce_e_binary_non_term(3, elems) })]
    #[production(r"<B>", reduce_e_unary_non_term)]
    E(Box<NonTermKind>),
    #[production(r"TerminalKind::Int", reduce_b_non_term)]
    B(Terminal),
}

impl NonTerminalRepresentable for NonTerminal {
    type Terminal = Terminal;
}

#[test]
fn derived_macro_generator_should_parse_tokens_with_embedded_values() {
    let input = [
        Terminal::Int(10),
        Terminal::Plus,
        Terminal::Int(1),
        Terminal::Eof,
    ];
    let tokenizer = input.into_iter();

    let parse_tree = NonTerminal::parse_input(tokenizer);

    let expected = NonTerminal::E(Box::new(NonTermKind::Add(
        NonTerminal::E(Box::new(NonTermKind::Unary(NonTerminal::B(Terminal::Int(
            10,
        ))))),
        NonTerminal::B(Terminal::Int(1)),
    )));

    assert_eq!(parse_tree, Ok(expected));
}
