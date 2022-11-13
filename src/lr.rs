use std::{
    collections::{HashMap, HashSet},
    fmt::write,
};

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
    fn insert<T: Into<Token<'a>>>(&mut self, key: Symbol<'a>, token: T) -> bool {
        self.sets
            .get_mut(&key)
            .map(|token_set| token_set.insert(token.into()))
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

impl<'a> std::fmt::Display for SymbolTokenSet<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lines = self
            .sets
            .iter()
            .map(|(symbol, toks)| {
                let rhs = toks.iter().map(|tok| tok.to_string()).collect::<Vec<_>>();

                format!("{}: {}", symbol.as_ref(), rhs.join(", "))
            })
            .collect::<Vec<_>>();

        write!(f, "{}", lines.join("\n"))
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
        first_set.insert(*symbol, BuiltinTokens::Epsilon);
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

fn build_follow_set<'a>(grammar_table: &'a GrammarTable) -> SymbolTokenSet<'a> {
    let symbols = grammar_table.symbols().collect::<Vec<_>>();
    let tokens = grammar_table.tokens().collect::<Vec<_>>();
    let mut follow_set = SymbolTokenSet::new(&symbols);

    // add eof to root rule symbol
    let first_symbol = {
        grammar_table
            .rules()
            .next()
            .and_then(|first_rule| match first_rule.rhs.get(0) {
                Some(SymbolOrTokenRef::Symbol(idx)) => symbols.get(*idx).copied(),
                _ => None,
            })
    }
    .unwrap();

    follow_set.insert(first_symbol, BuiltinTokens::Eof);

    // set EOF on all symbols that are the last rhs item.
    let symbols_at_end_of_production =
        grammar_table
            .rules()
            .filter_map(|first_rule| match first_rule.rhs.last() {
                Some(SymbolOrTokenRef::Symbol(idx)) => symbols.get(*idx).copied(),
                _ => None,
            });
    for symbol in symbols_at_end_of_production {
        follow_set.insert(symbol, BuiltinTokens::Eof);
    }

    // populate the rest of the sets with any following terminals.
    for rhs in grammar_table.rules().map(|rule_ref| &rule_ref.rhs) {
        // get the symbol and position into rdx of each symbol.
        let symbols_in_rule = rhs.iter().enumerate().filter_map(|(idx, sotr)| match sotr {
            SymbolOrTokenRef::Symbol(symbol_ref) => Some((idx, symbols[*symbol_ref])),
            _ => None,
        });

        for (symbols_idx_into_rhs, symbol) in symbols_in_rule {
            if let Some(SymbolOrTokenRef::Token(token_ref)) = rhs.get(symbols_idx_into_rhs + 1) {
                let token = tokens[*token_ref];
                follow_set.insert(symbol, token);
            }
        }
    }

    follow_set
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
    fn first_set_returns_expected_values() {
        let grammar_table = load_grammar(TEST_GRAMMAR);

        assert!(grammar_table.is_ok());

        // safe to unwrap with assertion.
        let grammar_table = grammar_table.unwrap();

        let nullable_terms = find_nullable_nonterminals(&grammar_table);
        let first_sets = build_first_set(&grammar_table, &nullable_terms);

        let got = first_sets.sets.into_iter().collect::<Vec<_>>();
        let expected = vec![(
            Symbol::new("<parens>"),
            [Token::new("(")].into_iter().collect::<HashSet<_>>(),
        )];

        assert_eq!(expected, got)
    }

    #[test]
    fn follow_set_returns_expected_values() {
        let grammar_table = load_grammar(TEST_GRAMMAR);

        assert!(grammar_table.is_ok());

        // safe to unwrap with assertion.
        let grammar_table = grammar_table.unwrap();

        let follow_set = build_follow_set(&grammar_table);

        let got = follow_set.sets.into_iter().collect::<Vec<_>>();
        let expected = vec![(
            Symbol::new("<parens>"),
            [Token::new("("), Token::new(")"), Token::new("<$>")]
                .into_iter()
                .collect::<HashSet<_>>(),
        )];

        assert_eq!(expected, got)
    }
}
