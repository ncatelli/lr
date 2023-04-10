use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lr_core::{TerminalOrNonTerminal, TerminalRepresentable};
pub use lr_derive::Lr1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalKind {
    Plus,
    Minus,
    Star,
    Slash,
    Int,
    Eof,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Terminal {
    Plus,
    Minus,
    Star,
    Slash,
    Int(i64),
    Eof,
}

impl std::fmt::Display for Terminal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Plus => write!(f, "+"),
            Self::Minus => write!(f, "-"),
            Self::Star => write!(f, "*"),
            Self::Slash => write!(f, "/"),
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
            Self::Minus => Self::Repr::Minus,
            Self::Star => Self::Repr::Star,
            Self::Slash => Self::Repr::Slash,
            Self::Int(_) => Self::Repr::Int,
            Self::Eof => Self::Repr::Eof,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum MultiplicativeTermKind {
    Mul(NonTerminal, NonTerminal),
    Div(NonTerminal, NonTerminal),
    Unary(NonTerminal),
}

#[derive(Debug, PartialEq)]
pub enum AdditiveTermKind {
    Add(NonTerminal, NonTerminal),
    Sub(NonTerminal, NonTerminal),
    Unary(NonTerminal),
}

type TermOrNonTerm = TerminalOrNonTerminal<Terminal, NonTerminal>;

#[allow(unused)]
fn reduce_primary(elems: &mut Vec<TermOrNonTerm>) -> Result<NonTerminal, String> {
    if let Some(TermOrNonTerm::Terminal(term)) = elems.pop() {
        Ok(NonTerminal::Primary(term))
    } else {
        Err("expected terminal at top of stack in reducer.".to_string())
    }
}

#[allow(unused)]
fn reduce_expr_unary(elems: &mut Vec<TermOrNonTerm>) -> Result<NonTerminal, String> {
    if let Some(TermOrNonTerm::NonTerminal(nonterm)) = elems.pop() {
        Ok(NonTerminal::Expr(Box::new(nonterm)))
    } else {
        Err(format!(
            "expected non-terminal at top of stack in production 3 reducer.",
        ))
    }
}

#[allow(unused)]
fn reduce_multiplicative_unary(elems: &mut Vec<TermOrNonTerm>) -> Result<NonTerminal, String> {
    if let Some(TermOrNonTerm::NonTerminal(nonterm)) = elems.pop() {
        let non_term_kind = MultiplicativeTermKind::Unary(nonterm);

        Ok(NonTerminal::Multiplicative(Box::new(non_term_kind)))
    } else {
        Err(format!(
            "expected non-terminal at top of stack in production 3 reducer.",
        ))
    }
}

#[allow(unused)]
fn reduce_additive_unary(elems: &mut Vec<TermOrNonTerm>) -> Result<NonTerminal, String> {
    if let Some(TermOrNonTerm::NonTerminal(nonterm)) = elems.pop() {
        let non_term_kind = AdditiveTermKind::Unary(nonterm);

        Ok(NonTerminal::Additive(Box::new(non_term_kind)))
    } else {
        Err(format!(
            "expected non-terminal at top of stack in production 3 reducer.",
        ))
    }
}

#[allow(unused)]
fn reduce_multiplicative_binary(
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
            Terminal::Star => MultiplicativeTermKind::Mul(lhs, rhs),
            Terminal::Slash => MultiplicativeTermKind::Div(lhs, rhs),
            // Dispatcher should never reach this block of code due to parser guarantees.
            _ => unreachable!(),
        };

        Ok(NonTerminal::Multiplicative(Box::new(non_term_kind)))
    } else {
        Err(err_msg)
    }
}

#[allow(unused)]
fn reduce_additive_binary(elems: &mut Vec<TermOrNonTerm>) -> Result<NonTerminal, String> {
    let optional_rhs = elems.pop();
    let optional_term = elems.pop();
    let optional_lhs = elems.pop();
    let err_msg = format!(
        "expected 3 elements at top of stack in production  reducer. got [{:?}, {:?}, {:?}]",
        &optional_lhs, &optional_term, &optional_rhs
    );

    // reversed due to popping elements
    if let [Some(TermOrNonTerm::NonTerminal(lhs)), Some(TermOrNonTerm::Terminal(op)), Some(TerminalOrNonTerminal::NonTerminal(rhs))] =
        [optional_lhs, optional_term, optional_rhs]
    {
        let non_term_kind = match op {
            Terminal::Plus => AdditiveTermKind::Add(lhs, rhs),
            Terminal::Minus => AdditiveTermKind::Sub(lhs, rhs),
            // Dispatcher should never reach this block of code due to parser guarantees.
            _ => unreachable!(),
        };

        Ok(NonTerminal::Additive(Box::new(non_term_kind)))
    } else {
        Err(err_msg)
    }
}

#[derive(Debug, Lr1, PartialEq)]
pub enum NonTerminal {
    #[goal(r"<Expr>", reduce_expr_unary)]
    #[production(r"<Additive>", reduce_expr_unary)]
    Expr(Box<NonTerminal>),
    #[production(r"<Additive> Terminal::Plus <Multiplicative>", reduce_additive_binary)]
    #[production(r"<Additive> Terminal::Minus <Multiplicative>", reduce_additive_binary)]
    #[production(r"<Multiplicative>", reduce_additive_unary)]
    Additive(Box<AdditiveTermKind>),
    #[production(r"<Multiplicative> Terminal::Star <Primary>", |elems| { reduce_multiplicative_binary(6, elems) })]
    #[production(r"<Multiplicative> Terminal::Slash <Primary>", |elems| { reduce_multiplicative_binary(7, elems) })]
    #[production(r"<Primary>", reduce_multiplicative_unary)]
    Multiplicative(Box<MultiplicativeTermKind>),
    #[production(r"Terminal::Int", reduce_primary)]
    Primary(Terminal),
}

fn parse_basic_expression(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple calculator expression parsing");

    group.bench_function("without tokenization", |b| {
        let token_stream = [
            Terminal::Int(10),
            Terminal::Slash,
            Terminal::Int(5),
            Terminal::Plus,
            Terminal::Int(1),
            Terminal::Eof,
        ];

        let expected = NonTerminal::Expr(Box::new(NonTerminal::Additive(Box::new(
            AdditiveTermKind::Add(
                NonTerminal::Additive(Box::new(AdditiveTermKind::Unary(
                    NonTerminal::Multiplicative(Box::new(MultiplicativeTermKind::Div(
                        NonTerminal::Multiplicative(Box::new(MultiplicativeTermKind::Unary(
                            NonTerminal::Primary(Terminal::Int(10)),
                        ))),
                        NonTerminal::Primary(Terminal::Int(5)),
                    ))),
                ))),
                NonTerminal::Multiplicative(Box::new(MultiplicativeTermKind::Unary(
                    NonTerminal::Primary(Terminal::Int(1)),
                ))),
            ),
        ))));

        let expected = Ok(expected);

        b.iter(|| {
            let parse_tree = lr_parse_input(black_box((&token_stream).iter().copied()));
            assert_eq!(&parse_tree, &expected);
        });
    });
}

fn parse_large_expression(c: &mut Criterion) {
    let mut group = c.benchmark_group("large expression");
    let token_stream = [Terminal::Int(10)]
        .into_iter()
        .chain(
            [
                [Terminal::Slash, Terminal::Int(5)],
                [Terminal::Plus, Terminal::Int(1)],
                [Terminal::Minus, Terminal::Int(2)],
                [Terminal::Star, Terminal::Int(6)],
            ]
            .into_iter()
            .cycle()
            .take(100)
            .flatten(),
        )
        .chain([Terminal::Eof])
        .collect::<Vec<_>>();

    group.bench_function("without tokenization", |b| {
        b.iter(|| {
            let parse_tree = lr_parse_input(black_box((&token_stream).iter().copied()));
            assert!(parse_tree.is_ok());
        });
    });
}

criterion_group!(benches, parse_basic_expression, parse_large_expression);
criterion_main!(benches);
