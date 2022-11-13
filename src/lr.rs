use std::collections::{HashMap, HashSet};

use crate::grammar::*;

#[derive(Debug, PartialEq, Eq)]
pub enum ParserGenErrorKind {
    Other,
}

impl std::fmt::Display for ParserGenErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Other => write!(f, "undefined load error"),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ParserGenError {
    kind: ParserGenErrorKind,
    data: Option<String>,
}

impl ParserGenError {
    pub fn new(kind: ParserGenErrorKind) -> Self {
        Self { kind, data: None }
    }

    pub fn with_data_mut(&mut self, data: String) {
        self.data = Some(data)
    }

    pub fn with_data(mut self, data: String) -> Self {
        self.with_data_mut(data);
        self
    }
}

impl std::fmt::Display for ParserGenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.data {
            Some(ctx) => write!(f, "{}: {}", &self.kind, ctx),
            None => write!(f, "{}", &self.kind),
        }
    }
}

#[derive(Debug, PartialEq)]
struct SymbolTokenSet<'a> {
    sets: HashMap<Symbol<'a>, HashSet<Token<'a>>>,
}

impl<'a> SymbolTokenSet<'a> {
    fn new<S: AsRef<[Symbol<'a>]>>(symbols: S) -> Self {
        let sets = symbols
            .as_ref()
            .iter()
            .fold(HashMap::new(), |mut acc, &symbol| {
                acc.insert(symbol, HashSet::new());
                acc
            });
        Self { sets }
    }

    /// Inserts a token into a symbol's set returning true if it already exists.
    fn insert(&mut self, key: Symbol<'a>, token: Token<'a>) -> bool {
        self.sets
            .get_mut(&key)
            .map(|token_set| token_set.insert(token))
            .unwrap_or(false)
    }

    /// Returns a bool representing if a token is set for a given symbol.
    fn contains_token(&self, key: &Symbol<'a>, token: &Token<'a>) -> bool {
        self.sets
            .get(key)
            .map(|token_sets| token_sets.contains(token))
            .unwrap_or(false)
    }

    /// sets the tokens for `lhs` to the union of `lhs` and `rhs`.
    fn union_of_sets(&mut self, lhs: Symbol<'a>, rhs: &Symbol<'a>) {
        // get all terminals from the rhs symbol
        let first_tokens_from_rhs_symbol = self.sets.get(rhs).cloned().unwrap_or_default();
        self.sets.entry(lhs).and_modify(|token_set| {
            for token in first_tokens_from_rhs_symbol {
                token_set.insert(token);
            }
        });
    }
}

pub struct LrParser;

fn build_first_set<'a>(
    grammar_table: &'a GrammarTable,
    nullable_nonterminals: &HashSet<Symbol<'a>>,
) -> SymbolTokenSet<'a> {
    let symbols = grammar_table.symbols().collect::<Vec<_>>();
    let tokens = grammar_table.tokens().collect::<Vec<_>>();
    let mut first_set = SymbolTokenSet::new(&symbols);

    // map nullable nonterminals to epsilon
    for symbol in nullable_nonterminals {
        first_set.insert(*symbol, Token::new(BuiltinTokens::Epsilon.as_token()));
    }

    // set the initial token for each production
    let initial_tokens_of_productions = grammar_table.rules().filter_map(|rule_ref| {
        let lhs_idx = rule_ref.lhs;
        let lhs_symbol = symbols[lhs_idx];

        if let Some(SymbolOrTokenRef::Token(idx)) = rule_ref.rhs.get(0) {
            // if the the first token in the pattern isn't epsilon, add it.
            let first_token = tokens[*idx];
            Some((lhs_symbol, first_token))
        } else {
            None
        }
    });

    // map initial tokens in each rule to their symbol
    for (symbol, first_token) in initial_tokens_of_productions {
        first_set.insert(symbol, first_token);
    }

    // set the initial token for each production
    for rule_ref in grammar_table.rules() {
        let lhs_idx = rule_ref.lhs;
        let lhs_symbol = symbols[lhs_idx];

        if let Some(SymbolOrTokenRef::Symbol(idx)) = rule_ref.rhs.get(0) {
            // get all terminals from the first symbol
            let first_rhs_symbol = symbols[*idx];
            first_set.union_of_sets(lhs_symbol, &first_rhs_symbol)
        }
    }

    first_set
}

fn find_nullable_nonterminals<'a>(grammar_table: &'a GrammarTable) -> HashSet<Symbol> {
    let symbols = grammar_table.symbols().collect::<Vec<_>>();
    let tokens = grammar_table.tokens().collect::<Vec<_>>();
    let mut nullable_nonterminal_productions = HashSet::new();

    let mut done = false;
    while !done {
        // assume done unless a change happens.
        done = true;
        for rule in grammar_table.rules() {
            let lhs_id = rule.lhs;
            let lhs = symbols[lhs_id];

            // validate that the production isn't already nullable
            if !nullable_nonterminal_productions.contains(&lhs) {
                let first_rhs_is_token = rule.rhs.get(0).and_then(|sotr| match sotr {
                    SymbolOrTokenRef::Symbol(_) => None,
                    SymbolOrTokenRef::Token(idx) => tokens.get(*idx),
                });
                if first_rhs_is_token == Some(&Token::new(BuiltinTokens::Epsilon.as_token())) {
                    nullable_nonterminal_productions.insert(lhs);
                    done = false
                } else {
                    // check that the production doesn't contain a token or is not nullable.
                    let all_nullable = rule.rhs.iter().any(|sotr| match sotr {
                        SymbolOrTokenRef::Symbol(idx) => {
                            let symbol = symbols.get(*idx).unwrap();
                            nullable_nonterminal_productions.contains(symbol)
                        }
                        SymbolOrTokenRef::Token(_) => false,
                    });

                    if all_nullable {
                        nullable_nonterminal_productions.insert(lhs);
                        done = false
                    }
                }
            }
        }
    }

    nullable_nonterminal_productions
}

/// Build a LR(1) parser from a given grammar.
pub(crate) fn build_lr_parser(grammar_table: &GrammarTable) -> Result<LrParser, ParserGenError> {
    let _nullable_nonterminal_productions = find_nullable_nonterminals(grammar_table);

    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_GRAMMAR: &str = "
; a comment
<parens> ::= ( <parens> )
<parens> ::= ( )
<parens> ::= <parens> ( )
<parens> ::= <parens> ( <parens> )
<parens> ::= ( ) <parens>
<parens> ::= ( <parens> ) <parens>
";

    #[test]
    fn first_sets_returns_expected_values() {
        let grammar_table = load_grammar(TEST_GRAMMAR);

        assert!(grammar_table.is_ok());

        // safe to unwrap with assertion.
        let grammar_table = grammar_table.unwrap();

        let nullable_terms = find_nullable_nonterminals(&grammar_table);
        let first_sets = build_first_set(&grammar_table, &nullable_terms);
        let got = first_sets
            .sets
            .iter()
            .map(|(symbol, toks)| (*symbol, toks.iter().copied().collect::<Vec<_>>()))
            .collect::<Vec<_>>();

        let expected = vec![(Symbol::new("<parens>"), vec![Token::new("(")])];
        assert_eq!(expected, got)
    }
}
