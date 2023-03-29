#[allow(unused)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Terminal {
    Epsilon,
    EoF,
    EndL,
    Plus,
    Star,
    Zero,
    One,
}

impl std::fmt::Display for Terminal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let repr = match self {
            Terminal::Epsilon => "<epsilon>",
            Terminal::EoF => "<$>",
            Terminal::EndL => "<endl>",

            Terminal::Plus => "+",
            Terminal::Star => "*",
            Terminal::Zero => "0",
            Terminal::One => "1",
        };

        write!(f, "{}", repr)
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum ENonTermKind {
    Mul(NonTerminal, NonTerminal),
    Add(NonTerminal, NonTerminal),
    Unary(NonTerminal),
}

#[derive(Debug, PartialEq)]
pub(crate) enum NonTerminal {
    E(Box<ENonTermKind>),
    B(Terminal),
}

#[derive(Debug)]
pub(crate) enum TermOrNonTerm {
    Terminal(Terminal),
    NonTerminal(NonTerminal),
}

pub(crate) struct ParseContext {
    state_stack: Vec<usize>,
    element_stack: Vec<TermOrNonTerm>,
}

impl ParseContext {
    pub(crate) fn push_state_mut(&mut self, state_id: usize) {
        self.state_stack.push(state_id)
    }

    pub(crate) fn pop_state_mut(&mut self) -> Option<usize> {
        self.state_stack.pop()
    }

    pub(crate) fn push_element_mut(&mut self, elem: TermOrNonTerm) {
        self.element_stack.push(elem)
    }

    #[allow(unused)]
    pub(crate) fn pop_element_mut(&mut self) -> Option<TermOrNonTerm> {
        self.element_stack.pop()
    }
}

impl Default for ParseContext {
    fn default() -> Self {
        Self {
            state_stack: vec![],
            element_stack: vec![],
        }
    }
}

fn lookup_goto(state: usize, non_term: &NonTerminal) -> Option<usize> {
    match (state, non_term) {
        (0, NonTerminal::E(_)) => Some(1),
        (0, NonTerminal::B(_)) => Some(2),
        (5, NonTerminal::B(_)) => Some(7),
        (6, NonTerminal::B(_)) => Some(8),

        _ => None,
    }
}

#[allow(unused)]
pub(crate) fn parse_input_stream<T: AsRef<[Terminal]>>(input: T) -> Result<NonTerminal, String> {
    use crate::lr::{Action, RuleId, StateId};

    let mut input = input.as_ref().iter().copied().peekable();
    let mut parse_ctx = ParseContext::default();
    parse_ctx.push_state_mut(0);

    loop {
        let current_state = parse_ctx
            .pop_state_mut()
            .ok_or_else(|| "state stack is empty".to_string())?;

        let next_term = input.peek().unwrap_or(&Terminal::EoF);
        let action = match (current_state, next_term) {
            (0, Terminal::Zero) => Ok(Action::Shift(StateId::unchecked_new(3))),
            (0, Terminal::One) => Ok(Action::Shift(StateId::unchecked_new(4))),

            (1, Terminal::EoF) => Ok(Action::Accept),
            (1, Terminal::Star) => Ok(Action::Shift(StateId::unchecked_new(5))),
            (1, Terminal::Plus) => Ok(Action::Shift(StateId::unchecked_new(6))),

            (2, Terminal::EoF) => Ok(Action::Reduce(RuleId::unchecked_new(4))),
            (2, Terminal::Star) => Ok(Action::Reduce(RuleId::unchecked_new(4))),
            (2, Terminal::Plus) => Ok(Action::Reduce(RuleId::unchecked_new(4))),

            (3, Terminal::EoF) => Ok(Action::Reduce(RuleId::unchecked_new(5))),
            (3, Terminal::Star) => Ok(Action::Reduce(RuleId::unchecked_new(5))),
            (3, Terminal::Plus) => Ok(Action::Reduce(RuleId::unchecked_new(5))),

            (4, Terminal::EoF) => Ok(Action::Reduce(RuleId::unchecked_new(6))),
            (4, Terminal::Star) => Ok(Action::Reduce(RuleId::unchecked_new(6))),
            (4, Terminal::Plus) => Ok(Action::Reduce(RuleId::unchecked_new(6))),

            (5, Terminal::Zero) => Ok(Action::Shift(StateId::unchecked_new(3))),
            (5, Terminal::One) => Ok(Action::Shift(StateId::unchecked_new(4))),

            (6, Terminal::Zero) => Ok(Action::Shift(StateId::unchecked_new(3))),
            (6, Terminal::One) => Ok(Action::Shift(StateId::unchecked_new(4))),

            (7, Terminal::EoF) => Ok(Action::Reduce(RuleId::unchecked_new(2))),
            (7, Terminal::Star) => Ok(Action::Reduce(RuleId::unchecked_new(2))),
            (7, Terminal::Plus) => Ok(Action::Reduce(RuleId::unchecked_new(2))),

            (8, Terminal::EoF) => Ok(Action::Reduce(RuleId::unchecked_new(3))),
            (8, Terminal::Star) => Ok(Action::Reduce(RuleId::unchecked_new(3))),
            (8, Terminal::Plus) => Ok(Action::Reduce(RuleId::unchecked_new(3))),

            _ => Err(format!(
                "unknown parser error with: {:?}",
                (current_state, input.peek())
            )),
        }?;

        match action {
            Action::Shift(next_state) => {
                // a shift should never occur on an eof making this safe to unwrap.
                let term = input.next().map(TermOrNonTerm::Terminal).unwrap();
                parse_ctx.push_element_mut(term);

                parse_ctx.push_state_mut(current_state);
                parse_ctx.push_state_mut(next_state.as_usize());
                Ok(())
            }
            Action::Reduce(reduce_to) => {
                let (rhs_len, non_term) = match reduce_to.as_usize() {
                    1 => (
                        1,
                        (|elems: &mut Vec<TermOrNonTerm>| {
                            if let Some(TermOrNonTerm::NonTerminal(nonterm)) = elems.pop() {
                                Ok(nonterm)
                            } else {
                                Err(format!(
                                    "expected non-terminal at top of stack in rule 1 reducer.",
                                ))
                            }
                        })(&mut parse_ctx.element_stack),
                    ),
                    2 => (
                        3,
                        (|elems: &mut Vec<TermOrNonTerm>| {
                            let optional_rhs = elems.pop();
                            let optional_term = elems.pop();
                            let optional_lhs = elems.pop();

                            if let [Some(TermOrNonTerm::NonTerminal(lhs)), Some(TermOrNonTerm::Terminal(Terminal::Star)), Some(TermOrNonTerm::NonTerminal(rhs))] =
                                [optional_lhs, optional_term, optional_rhs]
                            {
                                let non_term_kind = ENonTermKind::Mul(lhs, rhs);

                                Ok(NonTerminal::E(Box::new(non_term_kind)))
                            } else {
                                Err(format!(
                                    "expected 3 elements at top of stack in rule 1 reducer.",
                                ))
                            }
                        })(&mut parse_ctx.element_stack),
                    ),
                    3 => (
                        3,
                        (|elems: &mut Vec<TermOrNonTerm>| {
                            let optional_rhs = elems.pop();
                            let optional_term = elems.pop();
                            let optional_lhs = elems.pop();

                            // reversed due to popping elements
                            if let [Some(TermOrNonTerm::NonTerminal(lhs)), Some(TermOrNonTerm::Terminal(Terminal::Plus)), Some(TermOrNonTerm::NonTerminal(rhs))] =
                                [optional_lhs, optional_term, optional_rhs]
                            {
                                let non_term_kind = ENonTermKind::Add(lhs, rhs);

                                Ok(NonTerminal::E(Box::new(non_term_kind)))
                            } else {
                                Err(format!(
                                    "expected 3 elements at top of stack in rule 2 reducer.",
                                ))
                            }
                        })(&mut parse_ctx.element_stack),
                    ),
                    4 => (
                        1,
                        (|elems: &mut Vec<TermOrNonTerm>| {
                            if let Some(TermOrNonTerm::NonTerminal(nonterm)) = elems.pop() {
                                let non_term_kind = ENonTermKind::Unary(nonterm);

                                Ok(NonTerminal::E(Box::new(non_term_kind)))
                            } else {
                                Err(format!(
                                    "expected non-terminal at top of stack in rule 3 reducer.",
                                ))
                            }
                        })(&mut parse_ctx.element_stack),
                    ),
                    rule_id @ 5 | rule_id @ 6 => (
                        1,
                        (|elems: &mut Vec<TermOrNonTerm>| {
                            if let Some(TermOrNonTerm::Terminal(term)) = elems.pop() {
                                Ok(NonTerminal::B(term))
                            } else {
                                Err(format!(
                                    "expected terminal at top of stack in rule {} reducer.",
                                    rule_id
                                ))
                            }
                        })(&mut parse_ctx.element_stack),
                    ),
                    _ => (
                        0,
                        Err(format!(
                            "unable to reduce to rule {}.",
                            reduce_to.as_usize()
                        )),
                    ),
                };

                let non_term = non_term?;

                // peek at the last state before the nth element taken.
                let prev_state = {
                    let mut prev_state = parse_ctx.pop_state_mut();
                    for _ in 1..rhs_len {
                        prev_state = parse_ctx.pop_state_mut();
                    }
                    let prev_state =
                        prev_state.ok_or_else(|| "state stack is empty".to_string())?;
                    parse_ctx.push_state_mut(prev_state);
                    prev_state
                };

                let goto_state = lookup_goto(prev_state, &non_term).ok_or_else(|| {
                    format!(
                        "no goto state for non_terminal {:?} in state {}",
                        &non_term, current_state
                    )
                })?;

                parse_ctx.push_state_mut(goto_state);

                parse_ctx
                    .element_stack
                    .push(TermOrNonTerm::NonTerminal(non_term));

                Ok(())
            }
            Action::DeadState => Err(format!(
                "unexpected input {} for state {}",
                input.peek().unwrap_or(&Terminal::EoF),
                current_state
            )),
            Action::Accept => {
                let element = match parse_ctx.element_stack.len() {
                    1 => Ok(parse_ctx.element_stack.pop().unwrap()),
                    0 => Err("Reached accept state with empty stack".to_string()),
                    _ => Err(format!(
                        "Reached accept state with data on stack {:?}",
                        parse_ctx.element_stack,
                    )),
                }?;

                match element {
                    TermOrNonTerm::Terminal(term) => {
                        return Err(format!(
                            "top of stack was a terminal at accept state: {:?}",
                            term
                        ))
                    }
                    TermOrNonTerm::NonTerminal(nonterm) => return Ok(nonterm),
                }
            }
        }?;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::*;

    #[test]
    #[ignore = "testing for my purposes only"]
    fn should_build_expected_table_from_grammar() {
        let grammar = "
<E> ::= <E> * <B>
<E> ::= <E> + <B>
<E> ::= <B>
<B> ::= 0
<B> ::= 1";

        let grammar_table = grammar::load_grammar(grammar).unwrap();
        let _state_table = generate_table_from_grammar(GeneratorKind::Lr1, &grammar_table).unwrap();

        let input_stream = vec![Terminal::One, Terminal::Plus, Terminal::One];
        let parse_tree = parse_input_stream(input_stream).unwrap();

        // Add(1, 1)
        let expected = NonTerminal::E(Box::new(ENonTermKind::Add(
            NonTerminal::E(Box::new(ENonTermKind::Unary(NonTerminal::B(Terminal::One)))),
            NonTerminal::B(Terminal::One),
        )));

        assert_eq!(parse_tree, expected);
    }
}
