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
    fn union_of_sets(&mut self, lhs: Symbol<'a>, rhs: &Symbol<'a>) -> bool {
        let mut changed = false;

        // get all terminals from the rhs symbol
        let first_tokens_from_rhs_symbol = self.sets.get(rhs).cloned().unwrap_or_default();
        self.sets.entry(lhs).and_modify(|token_set| {
            for token in first_tokens_from_rhs_symbol {
                changed = token_set.insert(token);
            }
        });

        changed
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

impl<'a> AsRef<HashMap<Symbol<'a>, HashSet<Token<'a>>>> for SymbolTokenSet<'a> {
    fn as_ref(&self) -> &HashMap<Symbol<'a>, HashSet<Token<'a>>> {
        &self.sets
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
        let lhs_symbol = symbols[lhs_idx.as_usize()];

        if let Some(SymbolOrTokenRef::Token(idx)) = rule_ref.rhs.get(0) {
            // if the the first token in the pattern isn't epsilon, add it.
            let first_token = tokens[idx.as_usize()];
            Some((lhs_symbol, first_token))
        } else {
            None
        }
    });

    // map initial tokens in each rule to their symbol
    for (symbol, first_token) in initial_tokens_of_productions {
        first_set.insert(symbol, first_token);
    }

    let mut changed = true;

    while changed {
        changed = false;
        // set the initial token for each production
        for rule_ref in grammar_table.rules() {
            let lhs_idx = rule_ref.lhs;
            let lhs_symbol = symbols[lhs_idx.as_usize()];

            if let Some(SymbolOrTokenRef::Symbol(idx)) = rule_ref.rhs.get(0) {
                // get all terminals from the first symbol
                let first_rhs_symbol = symbols[idx.as_usize()];
                if first_set.union_of_sets(lhs_symbol, &first_rhs_symbol) {
                    changed = true;
                }
            }
        }
    }

    first_set
}

fn build_follow_set<'a>(
    grammar_table: &'a GrammarTable,
    first_sets: &'a SymbolTokenSet,
) -> SymbolTokenSet<'a> {
    let symbols = grammar_table.symbols().collect::<Vec<_>>();
    let tokens = grammar_table.tokens().collect::<Vec<_>>();
    let mut follow_set = SymbolTokenSet::new(&symbols);

    // 1) FOLLOW(S) = { $ }   // where S is the starting Non-Terminal
    follow_set.insert(Symbol::new("<goal>"), BuiltinTokens::Eof);

    let mut changed = true;
    while changed {
        changed = false;

        let symbol_and_rules_containing_it =
            grammar_table.symbols().enumerate().flat_map(|(sid, s)| {
                grammar_table.rules().filter_map(move |rule| {
                    if rule
                        .rhs
                        .contains(&SymbolOrTokenRef::Symbol(SymbolRef::new(sid)))
                    {
                        Some((s, rule))
                    } else {
                        None
                    }
                })
            });

        for (b, rule) in symbol_and_rules_containing_it {
            let symbol_ref = grammar_table.symbol_mapping(&b).unwrap();
            let rhs = &rule.rhs;
            let symbol_pos = rhs
                .iter()
                .position(|sotr| sotr == &SymbolOrTokenRef::Symbol(symbol_ref))
                // existence in this loop means it exists in the rule.
                .unwrap();

            let symbol_is_last_in_rhs = symbol_pos == rhs.len().saturating_sub(1);

            // 2) If A -> pBq is a production, where p, B and q are any grammar symbols,
            //    then everything in FIRST(q)  except Є is in FOLLOW(B).
            // and
            // 4) If A->pBq is a production and FIRST(q) contains Є,
            // then FOLLOW(B) contains { FIRST(q) – Є } U FOLLOW(A)
            if !symbol_is_last_in_rhs {
                let q = &rhs[symbol_pos + 1];
                match q {
                    SymbolOrTokenRef::Symbol(idx) => {
                        let q_symbol = symbols[idx.as_usize()];
                        let q_first_set = first_sets.sets.get(&q_symbol).unwrap();
                        let contains_epsilon =
                            q_first_set.contains(&Token::from(BuiltinTokens::Epsilon));

                        let q_first_set_sans_epsilon = q_first_set
                            .iter()
                            .filter(|&tok| tok != &Token::from(BuiltinTokens::Epsilon));

                        for &t in q_first_set_sans_epsilon {
                            if follow_set.insert(b, t) {
                                changed = true;
                            }
                        }

                        if contains_epsilon {
                            let a = &symbols[rule.lhs.as_usize()];

                            if follow_set.union_of_sets(b, a) {
                                changed = true;
                            }
                        }
                    }
                    SymbolOrTokenRef::Token(idx) => {
                        let q = tokens[idx.as_usize()];

                        if follow_set.insert(b, q) {
                            changed = true;
                        }
                    }
                }
            }

            // 3) If A->pB is a production, then everything in FOLLOW(A) is in FOLLOW(B).
            if symbol_is_last_in_rhs {
                let a = &symbols[rule.lhs.as_usize()];

                if follow_set.union_of_sets(b, a) {
                    changed = true;
                }
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
            let lhs = symbols[lhs_id.as_usize()];

            // validate that the production isn't already nullable
            if !nullable_nonterminal_productions.contains(&lhs) {
                let first_rhs_is_token = rule.rhs.get(0).and_then(|sotr| match sotr {
                    SymbolOrTokenRef::Symbol(_) => None,
                    SymbolOrTokenRef::Token(idx) => tokens.get(idx.as_usize()),
                });
                if first_rhs_is_token == Some(&Token::new(BuiltinTokens::Epsilon.as_token())) {
                    nullable_nonterminal_productions.insert(lhs);
                    done = false
                } else {
                    // check that the production doesn't contain a token or is not nullable.
                    let all_nullable = rule.rhs.iter().any(|sotr| match sotr {
                        SymbolOrTokenRef::Symbol(idx) => {
                            let symbol = symbols.get(idx.as_usize()).unwrap();
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct ItemRef<'a> {
    production: &'a RuleRef,
    dot_position: usize,
    lookahead: TokenRef,
}

impl<'a> ItemRef<'a> {
    fn new(production: &'a RuleRef, dot_position: usize, lookahead: TokenRef) -> Self {
        Self {
            production,
            dot_position,
            lookahead,
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
struct ItemSet<'a> {
    items: Vec<ItemRef<'a>>,
}

impl<'a> ItemSet<'a> {
    fn new(items: Vec<ItemRef<'a>>) -> Self {
        Self { items }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
struct ItemCollection<'a> {
    item_sets: Vec<ItemSet<'a>>,
}

fn closure<'a>(grammar_table: &'a GrammarTable, initial_item: ItemRef<'a>) -> ItemSet<'a> {
    let nullable_terms = find_nullable_nonterminals(grammar_table);
    let first_sets = build_first_set(grammar_table, &nullable_terms);
    let follow_sets = build_follow_set(grammar_table, &first_sets);

    let mut set = ItemSet::default();
    let mut curr = vec![initial_item.production.lhs];
    let mut visited = HashSet::new();

    let dot_position = initial_item.dot_position;

    while !curr.is_empty() {
        let mut next = vec![];

        for production_symbol_ref in curr.clone() {
            // mark the symbol visited
            visited.insert(production_symbol_ref);

            let production_symbol = grammar_table
                .symbols()
                .nth(production_symbol_ref.as_usize())
                .unwrap();

            let follow_set = follow_sets.sets.get(&production_symbol).unwrap();

            let rules_matching_production = grammar_table
                .rules()
                .filter(|rule| rule.lhs == production_symbol_ref);

            for rule in rules_matching_production {
                let next_after_dot = rule.rhs.get(dot_position).copied();

                match next_after_dot {
                    Some(SymbolOrTokenRef::Token(_)) => {
                        follow_set
                            .iter()
                            .filter_map(|tok| grammar_table.token_mapping(tok))
                            .map(|tok_ref| ItemRef::new(rule, dot_position, tok_ref))
                            .for_each(|item_ref| set.items.push(item_ref));
                    }
                    Some(SymbolOrTokenRef::Symbol(symbol_ref)) => {
                        // add the symbol for next iteration if it hasn't been
                        // visited yet.
                        if !visited.contains(&symbol_ref) {
                            next.push(symbol_ref);
                        }

                        follow_set
                            .iter()
                            .filter_map(|tok| grammar_table.token_mapping(tok))
                            .map(|tok_ref| ItemRef::new(rule, dot_position, tok_ref))
                            .for_each(|item_ref| set.items.push(item_ref));
                    }
                    None => {}
                }
            }
        }

        std::mem::swap(&mut curr, &mut next);
    }

    set
}

fn build_canonical_collection(grammar_table: &GrammarTable) -> ItemCollection {
    let first_rule = grammar_table.rules().next().unwrap();

    let mut i0 = ItemSet::default();
    let eof_token_ref = grammar_table
        .token_mapping(&Token::from(BuiltinTokens::Eof))
        .unwrap();
    i0.items.push(ItemRef::new(first_rule, 0, eof_token_ref));

    let _collection = ItemCollection::default();

    todo!()
}

/// Build a LR(1) parser from a given grammar.
pub(crate) fn build_lr_parser(grammar_table: &GrammarTable) -> Result<LrParser, ParserGenError> {
    let _nullable_nonterminal_productions = find_nullable_nonterminals(grammar_table);

    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_parse_set_with_nullable_nonterminal() {
        let grammar = "
<expr> ::= <expr> + <term>
<term> ::= <term> * <factor>
<expr> ::= <term>
<term> ::= <factor>
<factor> ::= <integer>
";
        let grammar_with_epsilon = "
 <term> ::= <integer>\n
<factor> ::= <epsilon>\n
        ";

        let grammar_table = load_grammar(grammar);
        assert!(grammar_table.is_ok());

        // safe to unwrap with assertion.
        let grammar_table = grammar_table.unwrap();

        // assert there are no nullable nonterminals terms.
        let no_nullable_nonterminals = find_nullable_nonterminals(&grammar_table)
            .into_iter()
            .next()
            .is_none();

        assert!(no_nullable_nonterminals);

        // check a grammar containing an nullable non_terminal.
        let grammar_table = load_grammar(grammar_with_epsilon);
        assert!(grammar_table.is_ok());

        // safe to unwrap with assertion.
        let grammar_table = grammar_table.unwrap();
        // assert there are no nullable nonterminals terms.

        let nullable_nonterminals = find_nullable_nonterminals(&grammar_table)
            .into_iter()
            .collect::<Vec<_>>();

        assert_eq!(nullable_nonterminals, vec![Symbol::new("<factor>")])
    }

    #[test]
    fn first_set_returns_expected_values() {
        let grammar = "<E> ::= <T>
<E> ::= ( <E> )
<T> ::= <integer>
<T> ::= + <T>
<T> ::= <T> + <integer>
";

        let grammar_table = load_grammar(grammar);

        // safe to unwrap with assertion.
        let grammar_table = grammar_table.unwrap();

        let nullable_terms = find_nullable_nonterminals(&grammar_table);
        let first_sets = build_first_set(&grammar_table, &nullable_terms);

        let mut got = first_sets.sets.into_iter().collect::<Vec<_>>();
        got.sort_by(|(a, _), (b, _)| a.as_ref().cmp(b.as_ref()));

        let expected = vec![
            (
                Symbol::new("<E>"),
                [Token::new("<integer>"), Token::new("+"), Token::new("(")]
                    .into_iter()
                    .collect::<HashSet<_>>(),
            ),
            (
                Symbol::new("<T>"),
                [Token::new("<integer>"), Token::new("+")]
                    .into_iter()
                    .collect::<HashSet<_>>(),
            ),
            (
                Symbol::new("<goal>"),
                [Token::new("<integer>"), Token::new("+"), Token::new("(")]
                    .into_iter()
                    .collect::<HashSet<_>>(),
            ),
        ];

        assert_eq!(&got, &expected)
    }

    #[test]
    fn follow_set_returns_expected_values() {
        let grammar = "<E> ::= <T>
<E> ::= ( <E> )
<T> ::= <integer>
<T> ::= + <T>
<T> ::= <T> + <integer>
";

        let grammar_table = load_grammar(grammar);

        assert!(grammar_table.is_ok());

        // safe to unwrap with assertion.
        let grammar_table = grammar_table.unwrap();

        let nullable_terms = find_nullable_nonterminals(&grammar_table);
        let first_sets = build_first_set(&grammar_table, &nullable_terms);

        let follow_set = build_follow_set(&grammar_table, &first_sets);

        let mut got = follow_set.sets.into_iter().collect::<Vec<_>>();
        got.sort_by(|(a, _), (b, _)| a.as_ref().cmp(b.as_ref()));

        let expected = vec![
            (
                Symbol::new("<E>"),
                [Token::new("<$>"), Token::new(")")]
                    .into_iter()
                    .collect::<HashSet<_>>(),
            ),
            (
                Symbol::new("<T>"),
                [Token::new("+"), Token::new(")"), Token::new("<$>")]
                    .into_iter()
                    .collect::<HashSet<_>>(),
            ),
            (
                Symbol::new("<goal>"),
                [Token::new("<$>")].into_iter().collect::<HashSet<_>>(),
            ),
        ];

        assert_eq!(got, expected)
    }

    #[test]
    fn closure_generates_expected_value_for_item() {
        let grammar = "<E> ::= <T>
<E> ::= ( <E> )
<T> ::= n
<T> ::= + <T>
<T> ::= <T> + n
";

        let grammar_table = load_grammar(grammar);

        assert!(grammar_table.is_ok());

        // safe to unwrap with assertion.
        let grammar_table = grammar_table.unwrap();
        let nullable_terms = find_nullable_nonterminals(&grammar_table);
        let first_sets = build_first_set(&grammar_table, &nullable_terms);

        let initial_rule = grammar_table.rules().next().unwrap();
        let eof = grammar_table
            .token_mapping(&Token::from(BuiltinTokens::Eof))
            .unwrap();

        let closure = closure(&grammar_table, ItemRef::new(initial_rule, 0, eof));

        let _formatted_closure = closure
            .items
            .into_iter()
            .map(|item_ref| {
                let rule_ref = item_ref.production;

                let dot_position = item_ref.dot_position;
                let production = grammar_table
                    .symbols()
                    .nth(rule_ref.lhs.as_usize())
                    .unwrap();
                let mut rhs = rule_ref
                    .rhs
                    .iter()
                    .filter_map(|sotr| match sotr {
                        SymbolOrTokenRef::Symbol(sym_ref) => grammar_table
                            .symbols()
                            .nth(sym_ref.as_usize())
                            .map(SymbolOrToken::Symbol),
                        SymbolOrTokenRef::Token(tok_ref) => grammar_table
                            .tokens()
                            .nth(tok_ref.as_usize())
                            .map(SymbolOrToken::Token),
                    })
                    .map(|sot| match sot {
                        SymbolOrToken::Symbol(s) => s.to_string(),
                        SymbolOrToken::Token(t) => t.to_string(),
                    })
                    .collect::<Vec<_>>();
                rhs.insert(dot_position, ".".to_string());

                let lookahead = grammar_table
                    .tokens()
                    .nth(item_ref.lookahead.as_usize())
                    .unwrap();

                format!("{} -> {} [{}]\n", &production, rhs.join(" "), lookahead)
            })
            .collect::<String>();

        //println!("{}", formatted_closure);
        println!("{:#?}", build_follow_set(&grammar_table, &first_sets))
    }
}
