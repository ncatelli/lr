use crate::{grammar::*, ordered_set};

/// Markers for the type of error encountered in table generation.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum TableGenErrorKind {
    /// A terminal that did not match the grammar has been encounter.
    UnknownTerminal,
}

impl std::fmt::Display for TableGenErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownTerminal => write!(f, "terminal is undefined"),
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

    pub(crate) fn with_data_mut<S: AsRef<str>>(&mut self, data: S) {
        let data = data.as_ref().to_string();

        self.data = Some(data)
    }

    pub(crate) fn with_data<S: AsRef<str>>(mut self, data: S) -> Self {
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

/// A wrapper type for Lr1 Parser tables.
pub(crate) struct Lr1;

impl LrTableGenerator for Lr1 {
    fn generate_table(grammar_table: &GrammarTable) -> Result<LrTable, TableGenError> {
        let collection = build_canonical_collection(grammar_table);

        build_table(grammar_table, &collection)
    }
}

fn initial_item_set(grammar_table: &GrammarTable) -> ItemSet {
    let eof_terminal_ref = grammar_table.eof_terminal_ref();

    grammar_table
        .productions()
        .take(1)
        .map(|first_production| ItemRef::new(first_production, 0, eof_terminal_ref))
        .collect::<ItemSet>()
}

type BetaSet = [SymbolRef];

#[derive(Debug, Hash, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ItemRef<'a> {
    production: &'a ProductionRef,
    dot_position: usize,
    lookahead: TerminalRef,
}

impl<'a> ItemRef<'a> {
    fn new(production: &'a ProductionRef, dot_position: usize, lookahead: TerminalRef) -> Self {
        Self {
            production,
            dot_position,
            lookahead,
        }
    }

    fn is_completed(&self) -> bool {
        self.dot_position == self.production.rhs.len()
    }

    /// Advances the dot position of a previous ItemRef, returning a new
    /// ItemRef if the position is not the last element in the Item.
    fn advance_dot_position(&self) -> Option<Self> {
        let is_completed = self.is_completed();

        if is_completed {
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
    fn symbol_after_dot(&self) -> Option<&SymbolRef> {
        let dot_position = self.dot_position;

        self.production.rhs.get(dot_position)
    }

    fn beta(&self) -> &BetaSet {
        if self.is_completed() {
            &self.production.rhs[self.dot_position..]
        } else {
            &self.production.rhs[(self.dot_position + 1)..]
        }
    }
}

/// ItemSet contains an ordered list of item references.
#[derive(Default, Hash, Debug, Clone, PartialEq, Eq)]
struct ItemSet<'a> {
    items: crate::ordered_set::OrderedSet<ItemRef<'a>>,
}

impl<'a> ItemSet<'a> {
    fn new(items: Vec<ItemRef<'a>>) -> Self {
        Self {
            items: items.into_iter().collect(),
        }
    }

    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// returns the length of the item set.
    #[allow(unused)]
    fn len(&self) -> usize {
        self.items.len()
    }

    fn human_readable_format(&self, grammar_table: &'a GrammarTable) -> String {
        let line_formatter = self.items.as_ref().iter().map(|item_ref| {
            let production_ref = item_ref.production;

            let dot_position = item_ref.dot_position;
            let production = grammar_table
                .non_terminals()
                .nth(production_ref.lhs.as_usize())
                .unwrap();
            let mut rhs = production_ref
                .rhs
                .iter()
                .filter_map(|sotr| match sotr {
                    SymbolRef::NonTerminal(sym_ref) => grammar_table
                        .non_terminals()
                        .nth(sym_ref.as_usize())
                        .map(Symbol::NonTerminal),
                    SymbolRef::Terminal(tok_ref) => grammar_table
                        .terminals()
                        .nth(tok_ref.as_usize())
                        .map(Symbol::Terminal),
                })
                .map(|sot| match sot {
                    Symbol::NonTerminal(s) => s.to_string(),
                    Symbol::Terminal(t) => t.to_string(),
                })
                .collect::<Vec<_>>();
            rhs.insert(dot_position, ".".to_string());

            let lookahead = grammar_table
                .terminals()
                .nth(item_ref.lookahead.as_usize())
                .unwrap();

            format!("{} -> {} [{}]\n", &production, rhs.join(" "), lookahead)
        });

        line_formatter.fold(String::new(), |mut acc, line| {
            acc.push_str(&line);
            acc
        })
    }
}

impl<'a> From<crate::ordered_set::OrderedSet<ItemRef<'a>>> for ItemSet<'a> {
    fn from(value: crate::ordered_set::OrderedSet<ItemRef<'a>>) -> Self {
        Self { items: value }
    }
}

impl<'a> From<Vec<ItemRef<'a>>> for ItemSet<'a> {
    fn from(value: Vec<ItemRef<'a>>) -> Self {
        Self::new(value)
    }
}

impl<'a> FromIterator<ItemRef<'a>> for ItemSet<'a> {
    fn from_iter<T: IntoIterator<Item = ItemRef<'a>>>(iter: T) -> Self {
        let set = iter.into_iter().collect::<Vec<_>>();
        Self::new(set)
    }
}

fn first(
    first_symbol_sets: &FirstSymbolSet,
    beta_sets: &[&[SymbolRef]],
) -> ordered_set::OrderedSet<TerminalRef> {
    let mut firsts = crate::ordered_set::OrderedSet::default();

    for set in beta_sets {
        let first_symbol_in_beta = set.first();

        match first_symbol_in_beta {
            // set is epsilon thus continue.
            None => {}
            Some(SymbolRef::Terminal(term_ref)) => {
                firsts.insert(*term_ref);
                // early exit
                return firsts;
            }
            Some(SymbolRef::NonTerminal(nt_ref)) => {
                if let Some(nt_firsts) = first_symbol_sets.as_ref().get(nt_ref) {
                    for &term_ref in nt_firsts.as_ref() {
                        firsts.insert(term_ref);
                    }
                    // early exit
                    return firsts;
                };
            }
        };
    }

    // this should only be reached if all first sets are nullable.
    firsts
}

fn follow(
    first_symbol_sets: &FirstSymbolSet,
    beta_sets: &[&[SymbolRef]],
) -> ordered_set::OrderedSet<TerminalRef> {
    beta_sets
        .iter()
        .filter_map(|set| {
            let firsts = first(first_symbol_sets, &[set]);
            // break the loop if the first set returns.
            if !firsts.is_empty() {
                Some(firsts)
            } else {
                None
            }
        })
        // Return the first beta set (or lookahead) with values.
        .next()
        // or return an empty set.
        .unwrap_or_default()
}

/// Generates the closure of a `ItemSet` using the following algorithm.
///
/// ```ignore
/// Closure(I)
/// repeat
///     for (each item [ A -> β1.Bβ2, a ] in I )
///         for (each production B -> β3 in G’)
///           for (each terminal b in FIRST(β2, a))
///             add [ B -> .β3 , b ] to set I;
/// until no more items are added to I;
/// return I;
/// ```
fn closure<'a>(grammar_table: &'a GrammarTable, i: ItemSet<'a>) -> ItemSet<'a> {
    // if the itemset is empty exit early
    if i.is_empty() {
        return i;
    }

    let mut set = i.items;

    // The offset into from which all sets after are new from the previous
    // iteration.
    let mut new_sets_idx = 0;
    // While set is still changing.
    while set.len() != new_sets_idx {
        // build a list of all new items and update the modified count.
        let new_items_in_set = set.as_ref()[new_sets_idx..].to_vec();
        new_sets_idx = set.len();

        for item in new_items_in_set {
            let lookahead = item.lookahead;
            let beta = item.beta();
            let symbol_after_dot_position = item.symbol_after_dot();

            // handles for terminals and end of production.
            let maybe_next_symbol_after_dot_is_non_terminal = match symbol_after_dot_position {
                Some(SymbolRef::NonTerminal(s)) => Some(*s),
                _ => None,
            };

            if let Some(next_nonterminal_after_dot) = maybe_next_symbol_after_dot_is_non_terminal {
                let follow_set = {
                    let lookahead_set = [SymbolRef::Terminal(lookahead)];
                    follow(grammar_table.first_set(), &[beta, &lookahead_set])
                };

                let matching_productions = grammar_table
                    .productions()
                    .filter(|production| production.lhs == next_nonterminal_after_dot);

                for production in matching_productions {
                    let new_item = follow_set
                        .iter()
                        .map(|lookahead| ItemRef::new(production, 0, *lookahead));

                    for new in new_item {
                        let _ = set.insert(new.clone());
                    }
                }
            }
        }
    }

    ItemSet::from(set)
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
fn goto<'a>(grammar_table: &'a GrammarTable, i: &ItemSet<'a>, x: SymbolRef) -> ItemSet<'a> {
    let symbols_after_dot = i.items.as_ref().iter().filter(|item_ref| {
        let symbol_after_dot = item_ref.symbol_after_dot();

        symbol_after_dot == Some(&x)
    });

    let j = symbols_after_dot
        .into_iter()
        .filter_map(|item| item.advance_dot_position())
        .collect();

    closure(grammar_table, j)
}

/// Contains the canonical collection of `ItemSet` states ordered by their
/// state id.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
struct ItemCollection<'a> {
    /// Stores a mapping of each child set spawned from the item set.
    item_sets: crate::ordered_set::OrderedSet<ItemSet<'a>>,
}

impl<'a> ItemCollection<'a> {
    fn states(&self) -> usize {
        self.item_sets.len()
    }

    /// inserts a value into the collection, returning `true` if the set does
    /// not contain the value.
    fn insert(&mut self, new_set: ItemSet<'a>) -> bool {
        self.item_sets.insert(new_set)
    }

    /// Returns the set id of a given set if it exists within the collection.
    fn id_from_set(&self, set: &ItemSet<'a>) -> Option<usize> {
        self.item_sets.position(set)
    }

    /// Prints a human readable representation of a given collection.
    #[allow(unused)]
    fn human_readable_format(&self, grammar_table: &'a GrammarTable) -> String {
        self.item_sets
            .as_ref()
            .iter()
            .enumerate()
            .fold(String::new(), |mut acc, (id, i)| {
                let formatted_line =
                    format!("\nS{}:\n{}", id, &i.human_readable_format(grammar_table));
                acc.push_str(&formatted_line);
                acc
            })
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
/// s0 ← closure ( [S’→S,EOF] )
/// S ← { s0 }
/// k ← 1
/// while (S is still changing )
/// ∀ sj ∈ S and ∀ x ∈ ( T ∪ NT )
///     sk ← goto(sj ,x)
///     record sj → sk on x
/// if sk ∉ S then
///     S ← S ∪ sk
///     k ← k + 1
/// ```
fn build_canonical_collection(grammar_table: &GrammarTable) -> ItemCollection<'_> {
    use crate::ordered_set::OrderedSet;

    let mut collection = ItemCollection::default();

    let initial_item_set = initial_item_set(grammar_table);
    let s0 = closure(grammar_table, initial_item_set);

    collection.insert(s0);

    let mut new_states_idx = 0_usize;

    while collection.states() != new_states_idx {
        let new_states_in_collection = collection.item_sets.as_ref()[new_states_idx..].to_vec();
        // bump the offset to account for only new states.
        new_states_idx = collection.states();

        for parent_state in new_states_in_collection.iter() {
            let symbols_after_dot = parent_state
                .items
                .as_ref()
                .iter()
                .filter_map(|item| item.symbol_after_dot().copied())
                .collect::<OrderedSet<_>>();

            for symbol_after_dot in symbols_after_dot {
                let new_state = goto(grammar_table, parent_state, symbol_after_dot);

                let _ = collection.insert(new_state);
            }
        }
    }

    collection
}

/// A wrapper type for annotating a production.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateId(usize);

impl StateId {
    /// Instantiates a new [StateId] from a reference id.
    ///
    /// # Safety
    ///
    /// Caller guarantees that the id usize corresponds to a valid state in
    /// the parse table.
    pub fn unchecked_new(id: usize) -> Self {
        StateId(id)
    }

    pub fn as_usize(&self) -> usize {
        self.0
    }
}

impl From<StateId> for usize {
    fn from(value: StateId) -> Self {
        value.as_usize()
    }
}

/// A wrapper type for annotating a prouduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionId(usize);

impl ProductionId {
    /// Instantiates a new [ProductionId] from a reference id.
    ///
    /// # Safety
    ///
    /// Caller guarantees that the id usize corresponds to a valid production
    /// id in the corresponding grammar.
    pub fn unchecked_new(id: usize) -> Self {
        ProductionId(id)
    }

    pub fn as_usize(&self) -> usize {
        self.0
    }
}

impl From<ProductionId> for usize {
    fn from(value: ProductionId) -> Self {
        value.as_usize()
    }
}

/// Represents one of 4 valid actions for the action table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// The goal state has been reached and a parse can be accepted.
    Accept,
    /// Shift the input on to the terminal stream.
    Shift(StateId),
    /// Reduce the production to a previous state and type.
    Reduce(ProductionId),
    /// No further actions for the parse.
    DeadState,
}

impl Default for Action {
    fn default() -> Self {
        Self::DeadState
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Goto {
    State(usize),
    DeadState,
}

impl Default for Goto {
    fn default() -> Self {
        Self::DeadState
    }
}

#[derive(Debug)]
pub struct LrTable {
    pub states: usize,
    pub goto: Vec<Vec<Goto>>,
    pub action: Vec<Vec<Action>>,
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
    pub fn human_readable_format(&self, grammar_table: &GrammarTable) -> String {
        const DEAD_STATE_STR: &str = " ";

        let left_side_padding = 8;
        let row_header = grammar_table
            .terminals()
            .map(|t| t.to_string())
            .chain(
                grammar_table
                    .non_terminals()
                    // skip the goal non-terminal
                    .skip(1)
                    .map(|s| s.to_string()),
            )
            .fold(String::new(), |mut acc, t_or_s_str_repr| {
                let formatted_symbol = format!("{: >10}", t_or_s_str_repr);
                acc.push_str(&formatted_symbol);
                acc
            });
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
            .map(|curr_state| {
                let action_row = self.action.iter().map(|col| {
                    col.get(curr_state)
                        .map(|a| match a {
                            Action::Accept => format!("{: >10}", "accept"),
                            Action::Shift(id) => format!("{: >10}", format!("s{}", id.as_usize())),
                            // productions are 1-indexed, when pretty printed
                            Action::Reduce(id) => {
                                format!("{: >10}", format!("r{}", id.as_usize() + 1))
                            }
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
/// ∀ set sx ∈ S
///     ∀ item i ∈ sx
///         if i is [A → β •ad,b] and goto(sx ,a) = sk , a ∈ T
///             then ACTION[x,a] ← “shift k”
///         else if i is [S’→S •, EOF]
///             then ACTION[x ,a] ← “accept”
///         else if i is [A →β •,a]
///             then ACTION[x,a] ← “reduce A → β”
///     ∀ n ∈ NT
///         if goto(sx ,n) = s k
///             then GOTO [x,n] ← k
/// ```
fn build_table<'a>(
    grammar_table: &'a GrammarTable,
    canonical_collection: &ItemCollection<'a>,
) -> Result<LrTable, TableGenError> {
    let terminals = grammar_table.terminals().collect::<Vec<_>>();
    let eof_terminal_ref = grammar_table.eof_terminal_ref();
    let eof_terminal = terminals[eof_terminal_ref.as_usize()];

    let mut goto_table: Vec<Vec<Goto>> = vec![
        vec![Goto::default(); canonical_collection.states()];
        grammar_table.non_terminals().count()
    ];
    let mut action_table: Vec<Vec<Action>> =
        vec![
            vec![Action::default(); canonical_collection.states()];
            grammar_table.terminals().count()
        ];

    for (x, sx) in canonical_collection.item_sets.as_ref().iter().enumerate() {
        let items = sx.items.as_ref();

        for i in items {
            let symbol_after_dot = i.symbol_after_dot().copied();

            // shift is symbol is both a terminal and not the end of input.
            if let Some(SymbolRef::Terminal(a)) = symbol_after_dot {
                let sk = goto(grammar_table, sx, SymbolRef::Terminal(a));
                let k = canonical_collection.id_from_set(&sk);

                if let Some(k) = k {
                    action_table[a.as_usize()][x] = Action::Shift(StateId::unchecked_new(k));
                    continue;
                }
            }

            let lookahead_terminal_ref = &i.lookahead;
            let lookahead_terminal = terminals
                .get(lookahead_terminal_ref.as_usize())
                .ok_or_else(|| {
                    TableGenError::new(TableGenErrorKind::UnknownTerminal)
                        .with_data(format!("{}", &i.lookahead))
                })?;

            // if it's the start action, accept
            let is_goal = Some(i.production.lhs)
                == grammar_table
                    .non_terminal_mapping(&NonTerminal::from(BuiltinNonTerminals::Goal));
            let is_goal_acceptor =
                is_goal && symbol_after_dot.is_none() && (*lookahead_terminal == eof_terminal);

            if is_goal_acceptor {
                let a = eof_terminal_ref;

                // Safe to assign without checks due all indexes being derived from known states.
                action_table[a.as_usize()][x] = Action::Accept;

            // else if i is [A →β •,a]
            //     then ACTION[x,a] ← “reduce A → β”
            } else if symbol_after_dot.is_none() {
                let a = lookahead_terminal_ref;

                // Generate a proudction from the current productions. for
                // matching the last state.
                let parsed_production_lhs = i.production.lhs;
                let parsed_production_rhs = &i.production.rhs[0..i.dot_position];
                let production_from_stack =
                    ProductionRef::new(parsed_production_lhs, parsed_production_rhs.to_vec())
                        .unwrap();

                // TODO: Store the ProductionId on the production.
                let production_id = grammar_table
                    .productions()
                    .position(|production| production == &production_from_stack);

                if let Some(production_id) = production_id {
                    action_table[a.as_usize()][x] =
                        Action::Reduce(ProductionId::unchecked_new(production_id));
                };
            }
        }

        // ∀ n ∈ NT
        //     if goto(sx ,n) = s k
        //         then GOTO [x,n] ← k
        let nt = grammar_table
            .non_terminals()
            .skip(1)
            .flat_map(|s| grammar_table.non_terminal_mapping(&s));

        let k_sets = nt.filter_map(|n| {
            let sk = goto(grammar_table, sx, SymbolRef::NonTerminal(n));

            // find the first set that contains all elements of sk.
            canonical_collection
                .item_sets
                .as_ref()
                .iter()
                .position(|sx| sk.items.as_ref().starts_with(sx.items.as_ref()))
                .map(|k| (n.as_usize(), k))
        });

        for (n, k) in k_sets {
            goto_table[n][x] = Goto::State(k);
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

    use std::collections::{HashMap, HashSet};

    const TEST_GRAMMAR: &str = "
<E> ::= <T> - <E>
<E> ::= <T>
<T> ::= <F> * <T>
<T> ::= <F>
<F> ::= n
";

    #[test]
    fn first_set_returns_expected_values() {
        let grammar = "<E> ::= <T>
<E> ::= ( <E> )
<T> ::= 0
<T> ::= + <T>
<T> ::= <T> + 0
";

        let grammar_table = load_grammar(grammar).unwrap();

        let terminals = grammar_table.terminals().collect::<Vec<_>>();
        let nonterminals = grammar_table.non_terminals().collect::<Vec<_>>();

        let mut got = grammar_table
            .first_set()
            .iter()
            .map(|(nt, terms)| {
                let terms = terms
                    .iter()
                    .filter_map(|t| terminals.get(t.as_usize()).cloned())
                    .collect::<HashSet<_>>();

                (nonterminals[nt.as_usize()], terms)
            })
            .collect::<Vec<_>>();
        got.sort_by(|(a, _), (b, _)| a.as_ref().cmp(b.as_ref()));

        let expected = vec![
            (
                NonTerminal::from(BuiltinNonTerminals::Goal),
                [Terminal::new("0"), Terminal::new("+"), Terminal::new("(")]
                    .into_iter()
                    .collect::<HashSet<_>>(),
            ),
            (
                NonTerminal::new("<E>"),
                [Terminal::new("0"), Terminal::new("+"), Terminal::new("(")]
                    .into_iter()
                    .collect::<HashSet<_>>(),
            ),
            (
                NonTerminal::new("<T>"),
                [Terminal::new("0"), Terminal::new("+")]
                    .into_iter()
                    .collect::<HashSet<_>>(),
            ),
        ];

        assert_eq!(&got, &expected)
    }

    #[test]
    fn closure_generates_expected_value_for_itemset() {
        let grammar = TEST_GRAMMAR;

        let grammar_table = load_grammar(grammar);
        let grammar_table = grammar_table.unwrap();

        let s0 = initial_item_set(&grammar_table);
        let s0 = closure(&grammar_table, s0);
        assert_eq!(s0.len(), 10, "{}", {
            let mut collection = ItemCollection::default();
            collection.insert(s0.clone());
            collection.human_readable_format(&grammar_table)
        });

        let s0_human_readable_repr = s0.human_readable_format(&grammar_table);
        for lookahead in ["<$>", "-", "*"] {
            let closure = format!("<F> -> . n [{}]", lookahead);
            assert!(
                s0_human_readable_repr.contains(closure.as_str()),
                "{}",
                s0_human_readable_repr
            );
        }
    }

    #[test]
    fn goto_generates_expected_value_for_itemset() {
        let grammar = TEST_GRAMMAR;
        let grammar_table = load_grammar(grammar).unwrap();

        let initial_item_set = initial_item_set(&grammar_table);
        let s0 = closure(&grammar_table, initial_item_set);
        assert_eq!(s0.len(), 10);

        let mut symbols_after_dot = {
            let mut symbol_after_dot = s0
                .items
                .as_ref()
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
    fn build_canonical_collection_generates_expected_states() {
        let grammar = "
<E> ::= <T> - <E>
<E> ::= <T>
<T> ::= <F> * <T>
<T> ::= <F>
<F> ::= n";
        let grammar_table = load_grammar(grammar).unwrap();

        let collection = build_canonical_collection(&grammar_table);
        assert_eq!(
            collection.states(),
            9,
            "{}",
            collection.human_readable_format(&grammar_table)
        );

        let expected_productions_per_state = [10, 1, 2, 4, 3, 10, 9, 1, 2];
        let state_productions_assertion_tuples = collection
            .item_sets
            .as_ref()
            .iter()
            .map(|state| state.len())
            .enumerate()
            .zip(expected_productions_per_state);
        for ((sid, items_in_state), expected_items) in state_productions_assertion_tuples {
            assert_eq!((sid, items_in_state), (sid, expected_items))
        }
    }

    #[derive(Default)]
    struct GrammarTestCase<'a> {
        grammar_table: &'a str,
        expected_first_set_pairings: HashMap<&'a str, usize>,
        expected_states: Option<usize>,
        state_production_cnt_assertions: HashMap<usize, usize>,
    }

    impl<'a> GrammarTestCase<'a> {
        fn with_grammar(mut self, grammar: &'a str) -> Self {
            self.grammar_table = grammar;
            self
        }

        fn with_expected_first_set_pair(
            mut self,
            non_terminal: &'a str,
            first_set_cnt: usize,
        ) -> Self {
            self.expected_first_set_pairings
                .insert(non_terminal, first_set_cnt);
            self
        }

        fn with_expected_states_cnt(mut self, state_cnt: usize) -> Self {
            self.expected_states = Some(state_cnt);
            self
        }

        fn with_expected_state_production_assertion(
            mut self,
            state: usize,
            expected_productions: usize,
        ) -> Self {
            self.state_production_cnt_assertions
                .insert(state, expected_productions);
            self
        }
    }

    impl<'a> GrammarTestCase<'a> {
        fn test(&self) {
            let grammar_table = load_grammar(self.grammar_table).unwrap();
            let first_sets = grammar_table.first_set();

            for (&nt, &expected_terms) in self.expected_first_set_pairings.iter() {
                let key = grammar_table
                    .non_terminal_mapping(&NonTerminal::from(nt))
                    .unwrap();
                let first_set = first_sets.as_ref().get(&key).unwrap();
                assert_eq!(first_set.len(), expected_terms);
            }

            let initial_item_set = initial_item_set(&grammar_table);

            assert_eq!(initial_item_set.len(), 1);

            let collection = build_canonical_collection(&grammar_table);
            if let Some(expected_states) = self.expected_states {
                assert_eq!(
                    collection.states(),
                    expected_states,
                    "{}",
                    collection.human_readable_format(&grammar_table)
                )
            };

            for (state_id, expected_productions) in self.state_production_cnt_assertions.iter() {
                let maybe_item_set = collection.item_sets.as_ref().get(*state_id);
                assert!(maybe_item_set.is_some());

                let item_set = maybe_item_set.unwrap();
                assert_eq!(item_set.len(), *expected_productions, "{}", {
                    let mut collection = ItemCollection::default();
                    collection.insert(item_set.clone());
                    collection.human_readable_format(&grammar_table)
                });
            }
        }
    }

    #[test]
    fn should_correctly_generate_closure_sets_for_recursive_grammar() {
        let test_case_1 = GrammarTestCase::default()
            .with_grammar(
                "
<Expression> ::= <Primary>
<Primary> ::= Token::Identifier  
<Primary> ::= <Constant>
<Primary> ::= Token::StringLiteral
<Primary> ::= Token::LeftParen <Expression> Token::RightParen
<Constant> ::= Token::IntegerConstant
<Constant> ::= Token::CharacterConstant
<Constant> ::= Token::FloatingConstant
",
            )
            .with_expected_first_set_pair("<*>", 6)
            .with_expected_first_set_pair("<Expression>", 6)
            .with_expected_first_set_pair("<Primary>", 6)
            .with_expected_first_set_pair("<Constant>", 3)
            .with_expected_state_production_assertion(0, 9)
            .with_expected_state_production_assertion(15, 9)
            .with_expected_states_cnt(22);

        let test_case_2 = GrammarTestCase::default()
            .with_grammar(
                "
<E> ::= <E> * <B>
<E> ::= <E> + <B> 
<E> ::= <B>
<B> ::= 0 
<B> ::= 1
        ",
            )
            .with_expected_first_set_pair("<*>", 2)
            .with_expected_first_set_pair("<E>", 2)
            .with_expected_first_set_pair("<B>", 2)
            .with_expected_state_production_assertion(0, 16)
            .with_expected_states_cnt(9);

        for test_case in [test_case_1, test_case_2] {
            test_case.test()
        }
    }

    #[test]
    fn should_correctly_generate_closure_sets_for_larger_grammar() {
        GrammarTestCase::default()
            .with_grammar(
                "
                <Expression> ::= <Assignment>
                <Assignment> ::= <Conditional>
                <Conditional> ::= <LogicalOr>
                <LogicalOr> ::= <LogicalAnd>
                <LogicalAnd> ::= <InclusiveOr>
                <InclusiveOr> ::= <ExclusiveOr>
                <ExclusiveOr> ::= <And>
                <And> ::= <Equality>
                <Equality> ::= <Relational>
                <Relational> ::= <Shift>
                <Shift> ::= <Additive>
                <Additive> ::= <Multiplicative>
                <Multiplicative> ::= <Cast>
                <Cast> ::= <Unary>
                <Unary> ::= <Postfix>
                <Unary> ::= Token::PlusPlus <Unary>
                <Unary> ::= Token::MinusMinus <Unary>
                <Unary> ::= <UnaryOperator> <Cast>
                <UnaryOperator> ::= Token::Ampersand
                <UnaryOperator> ::= Token::Star
                <UnaryOperator> ::= Token::Plus
                <UnaryOperator> ::= Token::Minus
                <UnaryOperator> ::= Token::Tilde
                <UnaryOperator> ::= Token::Bang
                <Postfix> ::= <Primary>
                <Postfix> ::= <Postfix> Token::LeftBracket <Expression> Token::RightBracket
                <Postfix> ::= <Postfix> Token::LeftParen Token::RightParen
                <Postfix> ::= <Postfix> Token::LeftParen <ArgumentExpressionList> Token::RightParen
                <Postfix> ::= <Postfix> Token::Dot Token::Identifier
                <Postfix> ::= <Postfix> Token::Arrow Token::Identifier
                <Postfix> ::= <Postfix> Token::PlusPlus
                <Postfix> ::= <Postfix> Token::MinusMinus
                <ArgumentExpressionList> ::= <Assignment>
                <Primary> ::= Token::Identifier
                <Primary> ::= <Constant>
                <Primary> ::= Token::StringLiteral
                <Primary> ::= Token::LeftParen <Expression> Token::RightParen
                <Constant> ::= Token::IntegerConstant
                <Constant> ::= Token::CharacterConstant
                <Constant> ::= Token::FloatingConstant
                ",
            )
            .with_expected_state_production_assertion(0, 208)
            .with_expected_states_cnt(141)
            .test();
    }
}
