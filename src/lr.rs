use std::collections::{HashMap, HashSet};

use crate::grammar::*;

/// Markers for the type of error encountered in table generation.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum TableGenErrorKind {
    /// A token that did not match the grammar has been encounter.
    UnknownToken,
}

impl std::fmt::Display for TableGenErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownToken => write!(f, "token is undefined"),
        }
    }
}

/// Represents errors that can occur in the table generation process.
#[derive(Debug, PartialEq, Eq)]
pub struct TableGenError {
    kind: TableGenErrorKind,
    data: Option<String>,
}

impl TableGenError {
    pub(crate) fn new(kind: TableGenErrorKind) -> Self {
        Self { kind, data: None }
    }

    pub(crate) fn with_data_mut(&mut self, data: String) {
        self.data = Some(data)
    }

    pub(crate) fn with_data(mut self, data: String) -> Self {
        self.with_data_mut(data);
        self
    }
}

impl std::fmt::Display for TableGenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.data {
            Some(ctx) => write!(f, "{}: {}", &self.kind, ctx),
            None => write!(f, "{}", &self.kind),
        }
    }
}

/// Exposes a trait for generating an LR table from a grammar.
pub(crate) trait LrTableGenerator {
    fn generate_table(grammar_table: &GrammarTable) -> Result<LrTable, TableGenError>;
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

    /// Sets the tokens for `lhs` to the union of `lhs` and `rhs`.
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

/// A wrapper type for Lr1 Parser tables.
pub(crate) struct Lr1;

impl LrTableGenerator for Lr1 {
    fn generate_table(grammar_table: &GrammarTable) -> Result<LrTable, TableGenError> {
        let collection = build_canonical_collection(grammar_table);

        build_table(grammar_table, &collection)
    }
}

fn initial_item_set(grammar_table: &GrammarTable) -> ItemSet {
    let eof_token_ref = grammar_table.builtin_token_mapping(&BuiltinTokens::Eof);
    grammar_table
        .rules()
        .take(1)
        .map(|first_rule| ItemRef::new(first_rule, 0, eof_token_ref))
        .collect::<ItemSet>()
}

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
    follow_set.insert(Symbol::from(BuiltinSymbols::Goal), BuiltinTokens::Eof);

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
            //    then everything in FIRST(q)  except ?? is in FOLLOW(B).
            // and
            // 4) If A->pBq is a production and FIRST(q) contains ??,
            // then FOLLOW(B) contains { FIRST(q) ??? ?? } U FOLLOW(A)
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

fn find_nullable_nonterminals(grammar_table: &GrammarTable) -> HashSet<Symbol> {
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

#[derive(Debug, Hash, Clone, PartialEq, Eq)]
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

    /// Advances the dot position of a previous ItemRef, returning a new
    /// ItemRef if the position is not the last element in the Item.
    fn to_next_dot_postion(&self) -> Option<Self> {
        let prev_rhs_len = self.production.rhs.len();
        let is_last = self.dot_position == prev_rhs_len;

        if is_last {
            None
        } else {
            let advanced = {
                let mut advanced = self.clone();
                advanced.dot_position = self.dot_position + 1;

                advanced
            };

            Some(advanced)
        }
    }

    /// Returns `Some(B)` in the production `[ A -> ?.B?, a ]`.
    fn symbol_after_dot(&self) -> Option<&SymbolOrTokenRef> {
        let dot_position = self.dot_position;

        self.production.rhs.get(dot_position)
    }
}

/// ItemSet contains an ordered list of item references.
#[derive(Default, Hash, Debug, Clone, PartialEq, Eq)]
struct ItemSet<'a> {
    items: Vec<ItemRef<'a>>,
}

impl<'a> ItemSet<'a> {
    fn new(items: Vec<ItemRef<'a>>) -> Self {
        Self { items }
    }

    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// returns the length of the item set.
    #[allow(unused)]
    fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns a boolean signifying if the passed item exists in the item set.
    fn contains(&self, item: &ItemRef<'a>) -> bool {
        self.items.contains(item)
    }

    fn human_readable_format(&self, grammar_table: &'a GrammarTable) -> String {
        self.items
            .iter()
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
            .collect::<String>()
    }
}

impl<'a> FromIterator<ItemRef<'a>> for ItemSet<'a> {
    fn from_iter<T: IntoIterator<Item = ItemRef<'a>>>(iter: T) -> Self {
        let set = iter.into_iter().collect::<Vec<_>>();
        Self::new(set)
    }
}

/// Generates the closure of a `ItemSet` using the following algorithm.
///
/// ```ignore
/// Closure(I)
/// repeat
///     for (each item [ A -> ?.B?, a ] in I )
///         for (each production B -> ? in G???)
///           for (each terminal b in FIRST(?a))
///             add [ B -> .? , b ] to set I;
/// until no more items are added to I;
/// return I;
/// ```
fn closure<'a>(grammar_table: &'a GrammarTable, i: ItemSet<'a>) -> ItemSet<'a> {
    // if the itemset is empty exit early
    if i.is_empty() {
        return i;
    }

    let symbols = grammar_table.symbols().collect::<Vec<_>>();
    let nullable_terms = find_nullable_nonterminals(grammar_table);
    let first_sets = build_first_set(grammar_table, &nullable_terms);
    let follow_sets = build_follow_set(grammar_table, &first_sets);

    let mut set = i.clone();

    let mut in_set = set.items.iter().fold(HashSet::new(), |mut acc, i| {
        acc.insert(i.clone());
        acc
    });

    let mut changed = true;
    while changed {
        changed = false;

        for item in set.items.clone() {
            let dot_position = item.dot_position;
            let items_after_dot = &item.production.rhs[dot_position..];
            let non_terminals = items_after_dot.iter().filter_map(|syms| match syms {
                SymbolOrTokenRef::Symbol(s) => Some(*s),
                SymbolOrTokenRef::Token(_) => None,
            });

            for non_terminal in non_terminals {
                let non_terminal_symbol = symbols[non_terminal.as_usize()];
                let follow_set = {
                    let mut follow_set = follow_sets
                        .sets
                        .get(&non_terminal_symbol)
                        .unwrap()
                        .iter()
                        .copied()
                        .collect::<Vec<_>>();
                    follow_set.sort();
                    follow_set
                };

                let matching_rules = grammar_table
                    .rules()
                    .filter(|rule| rule.lhs == non_terminal);

                for rule in matching_rules {
                    let new_item = follow_set
                        .iter()
                        .filter_map(|token| grammar_table.token_mapping(token))
                        .map(|lookahead| ItemRef::new(rule, 0, lookahead));

                    for new in new_item {
                        if in_set.insert(new.clone()) {
                            set.items.push(new);
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    set
}

/// Generates the goto of an `ItemSet` using the following algorithm.
///
/// ```ignore
/// Goto(I, X)
/// Initialise J to be the empty set;
/// for ( each item A -> ?.X?, a ] in I )
///     Add item A -> ?X.?, a ] to set J;   /* move the dot one step */
/// return Closure(J);    /* apply closure to the set */
/// ```
fn goto<'a>(grammar_table: &'a GrammarTable, i: &ItemSet<'a>, x: SymbolOrTokenRef) -> ItemSet<'a> {
    // reverse the initial set so it can be popped.
    let symbols_after_dot = i.items.iter().filter(|item_ref| {
        let symbol_after_dot = item_ref.symbol_after_dot();

        symbol_after_dot == Some(&x)
    });

    let j = symbols_after_dot
        .into_iter()
        .filter_map(|item| item.to_next_dot_postion())
        .collect();

    closure(grammar_table, j)
}

/// Contains the canonical collection of `ItemSet` states ordered by their
/// state id.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
struct ItemCollection<'a> {
    /// Stores a mapping of each child set spawned from the item set.
    item_sets: Vec<ItemSet<'a>>,
}

impl<'a> ItemCollection<'a> {
    fn states(&self) -> usize {
        self.item_sets.len()
    }

    /// Returns a boolean signifying a value is already in the set.
    fn contains(&self, new_set: &ItemSet<'a>) -> bool {
        self.item_sets.iter().any(|i| i == new_set)
    }

    /// inserts a value into the collection, returning `true` if the set does
    /// not contain the value.
    fn insert(&mut self, new_set: ItemSet<'a>) -> bool {
        let already_present = self.contains(&new_set);
        if !already_present {
            self.item_sets.push(new_set);
        }

        // if it's not already present, then a value has been inserted.
        !already_present
    }

    /// Returns the set id of a given set if it exists within the collection.
    fn id_from_set(&self, set: &ItemSet<'a>) -> Option<usize> {
        self.item_sets.iter().position(|s| s == set)
    }

    /// Prints a human readable representation of a given collection.
    #[allow(unused)]
    fn human_readable_format(&self, grammar_table: &'a GrammarTable) -> String {
        self.item_sets
            .iter()
            .enumerate()
            .map(|(id, i)| format!("\nS{}:\n{}", id, &i.human_readable_format(grammar_table)))
            .collect::<String>()
    }
}

impl<'a> IntoIterator for ItemCollection<'a> {
    type Item = ItemSet<'a>;

    type IntoIter = std::vec::IntoIter<ItemSet<'a>>;

    fn into_iter(self) -> Self::IntoIter {
        self.item_sets.into_iter()
    }
}

/// Constructs a canonical collection from a GrammarTable using the following algorithm.
///
/// ```ignore
/// s0 ??? closure ( [S??????S,EOF] )
/// S ??? {s0 }
/// k ??? 1
/// while (S is still changing )
/// ??? sj ??? S and ??? x ??? ( T ??? NT )
///     sk ??? goto(sj ,x)
///     record sj ??? sk on x
/// if sk ??? S then
///     S ??? S ??? sk
///     k ??? k + 1
/// ```
fn build_canonical_collection(grammar_table: &GrammarTable) -> ItemCollection {
    let mut collection = ItemCollection::default();

    let initial_item_set = initial_item_set(grammar_table);
    let s0 = closure(grammar_table, initial_item_set);

    let mut changing = collection.insert(s0);
    let mut new_states = vec![];

    while changing {
        changing = false;

        for parent_state in collection.item_sets.iter() {
            let symbols_after_dot = {
                let mut symbol_after_dot = parent_state
                    .items
                    .iter()
                    .filter_map(|item| item.symbol_after_dot().copied())
                    .collect::<Vec<_>>();
                symbol_after_dot.dedup();

                symbol_after_dot.into_iter()
            };

            for symbol_after_dot in symbols_after_dot {
                let new_state = goto(grammar_table, parent_state, symbol_after_dot);

                // Strips any items from the new state that exist in the parent
                // state.
                let non_duplicate_items = new_state
                    .items
                    .into_iter()
                    .filter(|item| !parent_state.contains(item));

                // Constructs the new state from the non duplicate item set, assigining a parent.
                let new_non_duplicate_state = non_duplicate_items.collect();

                if !collection.contains(&new_non_duplicate_state) {
                    new_states.push(new_non_duplicate_state);
                }
            }
        }

        for new_state in new_states {
            // if there are new states to insert, mark the collection as
            // changing.
            changing = collection.insert(new_state);
        }
        new_states = vec![];
    }

    collection
}

/// Represents one of 4 valid actions for the action table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Action {
    /// The goal state has been reached and a parse can be accepted.
    Accept,
    /// Shift the input on to the token stream.
    Shift(usize),
    /// Reduce the rule to a previous state and type.
    Reduce(usize),
    /// No further actions for the parse.
    DeadState,
}

impl Default for Action {
    fn default() -> Self {
        Self::DeadState
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Goto {
    State(usize),
    DeadState,
}

impl Default for Goto {
    fn default() -> Self {
        Self::DeadState
    }
}

#[derive(Debug)]
pub(crate) struct LrTable {
    states: usize,
    goto: Vec<Vec<Goto>>,
    action: Vec<Vec<Action>>,
}

impl LrTable {
    pub(crate) fn new(states: usize, goto: Vec<Vec<Goto>>, action: Vec<Vec<Action>>) -> Self {
        Self {
            states,
            goto,
            action,
        }
    }

    /// Outputs a human-readable representation of the grammar table.
    #[allow(unused)]
    fn human_readable_format(&self, grammar_table: &GrammarTable) -> String {
        const DEAD_STATE_STR: &str = " ";

        let left_side_padding = 8;
        let row_header = grammar_table
            .tokens()
            .map(|t| t.to_string())
            .chain(
                grammar_table
                    .symbols()
                    // skip the goal symbol
                    .skip(1)
                    .map(|s| s.to_string()),
            )
            .map(|t_or_s_str_repr| format!("{: >10}", t_or_s_str_repr))
            .collect::<String>();
        let table_width_without_left_side_padding = row_header.len();

        let first_row = format!(
            "{}{}",
            " ".chars()
                .cycle()
                .take(left_side_padding)
                .collect::<String>(),
            &row_header
        );
        let table_padding = format!(
            "{}{}",
            " ".chars()
                .cycle()
                .take(left_side_padding)
                .collect::<String>(),
            "-".chars()
                .cycle()
                .take(table_width_without_left_side_padding)
                .collect::<String>()
        );

        let rows = (0..self.states)
            .into_iter()
            .map(|curr_state| {
                let action_row = self.action.iter().map(|col| {
                    col.get(curr_state)
                        .map(|a| match a {
                            Action::Accept => format!("{: >10}", "accept"),
                            Action::Shift(id) => format!("{: >10}", format!("s{}", id)),
                            // rules are 1-indexed, when pretty printed
                            Action::Reduce(id) => format!("{: >10}", format!("r{}", id + 1)),
                            Action::DeadState => format!("{: >10}", DEAD_STATE_STR),
                        })
                        .unwrap_or_else(|| format!("{: >10}", ""))
                });
                let goto_row = self.goto.iter().skip(1).map(|col| {
                    col.get(curr_state)
                        .map(|g| match g {
                            Goto::State(id) => format!("{: >10}", id),
                            Goto::DeadState => format!("{: >10}", DEAD_STATE_STR),
                        })
                        .unwrap_or_else(|| format!("{: >10}", ""))
                });

                format!(
                    "{: >6} |{}{}",
                    curr_state,
                    action_row.collect::<String>(),
                    goto_row.collect::<String>()
                )
            })
            .collect::<Vec<_>>();

        [first_row, table_padding]
            .into_iter()
            .chain(rows)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Constructs action table from a canonical collection.
///
/// ```ignore
/// ??? set sx ??? S
///     ??? item i ??? sx
///         if i is [A ??? ?? ???ad,b] and goto(sx ,a) = sk , a ??? T
///             then ACTION[x,a] ??? ???shift k???
///         else if i is [S??????S ???, EOF]
///             then ACTION[x ,a] ??? ???accept???
///         else if i is [A ????? ???,a]
///             then ACTION[x,a] ??? ???reduce A ??? ?????
///     ??? n ??? NT
///         if goto(sx ,n) = s k
///             then GOTO [x,n] ??? k
/// ```
fn build_table<'a>(
    grammar_table: &'a GrammarTable,
    canonical_collection: &ItemCollection<'a>,
) -> Result<LrTable, TableGenError> {
    let tokens = grammar_table.tokens().collect::<Vec<_>>();
    let mut goto_table: Vec<Vec<Goto>> =
        vec![vec![Goto::default(); canonical_collection.states()]; grammar_table.symbols().count()];
    let mut action_table: Vec<Vec<Action>> =
        vec![
            vec![Action::default(); canonical_collection.states()];
            grammar_table.tokens().count()
        ];

    for (x, sx) in canonical_collection.item_sets.iter().enumerate() {
        let items = &sx.items;

        for i in items {
            let symbol_after_dot = i.symbol_after_dot().copied();
            let lookahead_token_ref = &i.lookahead;
            let lookahead_token = tokens.get(lookahead_token_ref.as_usize()).ok_or_else(|| {
                TableGenError::new(TableGenErrorKind::UnknownToken)
                    .with_data(format!("{}", &i.lookahead))
            })?;

            let is_goal = Some(i.production.lhs)
                == grammar_table.symbol_mapping(&Symbol::from(BuiltinSymbols::Goal));
            let is_goal_acceptor = is_goal
                && symbol_after_dot.is_none()
                && (*lookahead_token == Token::from(BuiltinTokens::Eof));

            // if not the last symbol and it's a token, setup a shift.
            if let Some(SymbolOrTokenRef::Token(a)) = symbol_after_dot {
                let sk = goto(grammar_table, sx, SymbolOrTokenRef::Token(a));
                let k = canonical_collection.id_from_set(&sk);

                if let Some(k) = k {
                    action_table[a.as_usize()][x] = Action::Shift(k);
                    continue;
                }
            }

            // if it's the start action, accept
            if is_goal_acceptor {
                // safe to unwrap, Eof builtin is guaranteed to exist.
                let a = grammar_table
                    .token_mapping(&Token::from(BuiltinTokens::Eof))
                    .unwrap();

                // Safe to assign without checks due all indexes being derived from known states.
                action_table[a.as_usize()][x] = Action::Accept;

            // else if i is [A ????? ???,a]
            //     then ACTION[x,a] ??? ???reduce A ??? ?????
            } else if symbol_after_dot.is_none() {
                let a = lookahead_token_ref;

                // Generate a rule from the current productions. for matching the last state.
                let parsed_production_lhs = i.production.lhs;
                let parsed_production_rhs = &i.production.rhs[0..i.dot_position];
                let production_from_stack =
                    RuleRef::new(parsed_production_lhs, parsed_production_rhs.to_vec()).unwrap();

                // TODO: Store the RuleId on the production.
                let rule_id = grammar_table
                    .rules()
                    .position(|rule| rule == &production_from_stack);

                if let Some(rule_id) = rule_id {
                    action_table[a.as_usize()][x] = Action::Reduce(rule_id);
                };
            }
        }

        // ??? n ??? NT
        //     if goto(sx ,n) = s k
        //         then GOTO [x,n] ??? k
        let nt = grammar_table
            .symbols()
            .skip(1)
            .flat_map(|s| grammar_table.symbol_mapping(&s));
        for n in nt {
            let sk = goto(grammar_table, sx, SymbolOrTokenRef::Symbol(n));

            // find the first set that contains all elements of sk.
            let k = canonical_collection
                .item_sets
                .iter()
                .position(|sx| sk.items.starts_with(&sx.items));

            if let Some(k) = k {
                goto_table[n.as_usize()][x] = Goto::State(k);
            }
        }
    }

    Ok(LrTable::new(
        canonical_collection.states(),
        goto_table,
        action_table,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_GRAMMAR: &str = "<E> ::= <T>
<E> ::= ( <E> )
<T> ::= n
<T> ::= + <T>
<T> ::= <T> + n
";

    #[test]
    fn should_parse_set_with_nullable_nonterminal() {
        let grammar = "
<expr> ::= <expr> + <term>
<term> ::= <term> * <factor>
<expr> ::= <term>
<term> ::= <factor>
<factor> ::= 0
";
        let grammar_with_epsilon = "
 <term> ::= 0\n
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
<T> ::= 0
<T> ::= + <T>
<T> ::= <T> + 0
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
                Symbol::from(BuiltinSymbols::Goal),
                [Token::new("0"), Token::new("+"), Token::new("(")]
                    .into_iter()
                    .collect::<HashSet<_>>(),
            ),
            (
                Symbol::new("<E>"),
                [Token::new("0"), Token::new("+"), Token::new("(")]
                    .into_iter()
                    .collect::<HashSet<_>>(),
            ),
            (
                Symbol::new("<T>"),
                [Token::new("0"), Token::new("+")]
                    .into_iter()
                    .collect::<HashSet<_>>(),
            ),
        ];

        assert_eq!(&got, &expected)
    }

    #[test]
    fn follow_set_returns_expected_values() {
        let grammar = TEST_GRAMMAR;
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
                Symbol::from(BuiltinSymbols::Goal),
                [Token::new("<$>")].into_iter().collect::<HashSet<_>>(),
            ),
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
        ];

        assert_eq!(got, expected)
    }

    #[test]
    fn closure_generates_expected_value_for_itemset() {
        let grammar = TEST_GRAMMAR;
        let grammar_table = load_grammar(grammar);

        assert!(grammar_table.is_ok());

        // safe to unwrap with assertion.
        let grammar_table = grammar_table.unwrap();

        let initial_rule = grammar_table.rules().next().unwrap();
        let eof = grammar_table
            .token_mapping(&Token::from(BuiltinTokens::Eof))
            .unwrap();

        let s0 = ItemSet::new(vec![ItemRef::new(initial_rule, 0, eof)]);
        let closure_res = closure(&grammar_table, s0);

        assert!(
            closure_res.len() == 14,
            "expected 14 items, got {}\n{}",
            closure_res.len(),
            closure_res.human_readable_format(&grammar_table)
        );

        let expected_lines = "
<*> -> . <E> [<$>]
<E> -> . <T> [<$>]
<E> -> . ( <E> ) [<$>]
<T> -> . n [)]
<T> -> . n [<$>]
<T> -> . + <T> [)]
<T> -> . + <T> [<$>]
<T> -> . <T> + n [)]
<T> -> . <T> + n [<$>]
<E> -> . <T> [)]
<E> -> . ( <E> ) [)]
<T> -> . n [+]
<T> -> . + <T> [+]
<T> -> . <T> + n [+]"
            .trim()
            .lines();

        let got = closure_res.human_readable_format(&grammar_table);
        for line in expected_lines {
            assert!(got.contains(line));
        }

        let grammar = "
<E> ::= <T> - <E>
<E> ::= <T>
<T> ::= <F> * <T>
<T> ::= <F>
<F> ::= n";
        let grammar_table = load_grammar(grammar);

        assert!(grammar_table.is_ok());

        // safe to unwrap with assertion.
        let grammar_table = grammar_table.unwrap();

        let s0 = initial_item_set(&grammar_table);
        let s0 = closure(&grammar_table, s0);
        assert_eq!(s0.len(), 10);
        assert!(s0
            .human_readable_format(&grammar_table)
            .contains("<F> -> . n [*]"));
    }

    #[test]
    fn goto_generates_expected_value_for_itemset() {
        let grammar = TEST_GRAMMAR;
        let grammar_table = load_grammar(grammar);

        assert!(grammar_table.is_ok());

        // safe to unwrap with assertion.
        let grammar_table = grammar_table.unwrap();

        let initial_item_set = initial_item_set(&grammar_table);
        let s0 = closure(&grammar_table, initial_item_set);
        assert_eq!(s0.len(), 14);

        let mut symbols_after_dot = {
            let mut symbol_after_dot = s0
                .items
                .iter()
                .filter_map(|item| item.symbol_after_dot().copied())
                .collect::<Vec<_>>();
            symbol_after_dot.dedup();

            symbol_after_dot.into_iter()
        };

        let symbol_after_dot = symbols_after_dot.next().unwrap();
        let s0 = goto(&grammar_table, &s0, symbol_after_dot);
        assert_eq!(s0.len(), 1,);
        assert_eq!(
            "<*> -> <E> . [<$>]\n",
            s0.human_readable_format(&grammar_table)
        );
    }

    #[test]
    fn collection_generates_expected_value_for_itemset() {
        let grammar = TEST_GRAMMAR;
        let grammar_table = load_grammar(grammar);

        // safe to unwrap with assertion.
        assert!(grammar_table.is_ok());
        let grammar_table = grammar_table.unwrap();

        let initial_item_set = initial_item_set(&grammar_table);
        let s0 = closure(&grammar_table, initial_item_set);
        assert_eq!(s0.len(), 14);

        let mut symbols_after_dot = {
            let mut symbol_after_dot = s0
                .items
                .iter()
                .filter_map(|item| item.symbol_after_dot().copied())
                .collect::<Vec<_>>();
            symbol_after_dot.dedup();

            symbol_after_dot.into_iter()
        };

        for (generation, expected_items) in [1, 5, 15, 3, 12]
            .into_iter()
            .enumerate()
            .map(|(gen, expected_rules)| (gen + 1, expected_rules))
        {
            let symbol_after_dot = symbols_after_dot.next().unwrap();
            let state = goto(&grammar_table, &s0, symbol_after_dot);
            assert_eq!(
                state.len(),
                expected_items,
                "\ngeneration: {}\ntoken: {}\n{}",
                generation,
                match symbol_after_dot {
                    SymbolOrTokenRef::Symbol(s) => {
                        grammar_table
                            .symbols()
                            .nth(s.as_usize())
                            .map(SymbolOrToken::Symbol)
                    }
                    SymbolOrTokenRef::Token(t) => grammar_table
                        .tokens()
                        .nth(t.as_usize())
                        .map(SymbolOrToken::Token),
                }
                .unwrap(),
                state.human_readable_format(&grammar_table)
            );
        }
    }

    #[test]
    fn build_canonical_collection_generates_expected_states() {
        let grammar = "
<E> ::= <T> - <E>
<E> ::= <T>
<T> ::= <F> * <T>
<T> ::= <F>
<F> ::= n";
        let grammar_table = load_grammar(grammar);

        // safe to unwrap with assertion.
        assert!(grammar_table.is_ok());
        let grammar_table = grammar_table.unwrap();

        let collection = build_canonical_collection(&grammar_table);
        assert_eq!(
            collection.states(),
            9,
            "{}",
            collection.human_readable_format(&grammar_table)
        );

        let expected_rules_per_state = [10, 1, 2, 4, 3, 10, 9, 1, 2];
        let state_rules_assertion_tuples = collection
            .item_sets
            .iter()
            .map(|state| state.len())
            .enumerate()
            .zip(expected_rules_per_state.into_iter());
        for ((sid, items_in_state), expected_items) in state_rules_assertion_tuples {
            assert_eq!((sid, items_in_state), (sid, expected_items))
        }
    }

    #[test]
    fn should_build_expected_table_from_grammar() {
        let grammar = "
<E> ::= <T> - <E>
<E> ::= <T>
<T> ::= <F> * <T>
<T> ::= <F>
<F> ::= 1";
        let grammar_table = load_grammar(grammar);

        // safe to unwrap with assertion.
        assert!(grammar_table.is_ok());
        let grammar_table = grammar_table.unwrap();

        let collection = build_canonical_collection(&grammar_table);
        let build_table_res = build_table(&grammar_table, &collection);

        assert!(build_table_res.is_ok());
        let table = build_table_res.unwrap();

        println!("{}", table.human_readable_format(&grammar_table))
    }
}
