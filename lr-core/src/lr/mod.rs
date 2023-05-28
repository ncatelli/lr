use std::collections::{HashMap, HashSet};

use crate::grammar::*;

mod ordered_set;

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

/// A mapping of non-terminal symbols to their corresponding terminal symbols.
#[derive(Debug, PartialEq)]
struct SymbolRefSet {
    sets: HashMap<NonTerminalRef, HashSet<TerminalRef>>,
}

impl SymbolRefSet {
    fn new<NT: AsRef<[NonTerminalRef]>>(non_terminals: NT) -> Self {
        let sets = non_terminals
            .as_ref()
            .iter()
            .fold(HashMap::new(), |mut acc, &non_terminal| {
                acc.insert(non_terminal, HashSet::new());
                acc
            });
        Self { sets }
    }

    /// Inserts a terminal into a non-terminals's set returning true if it already exists.
    fn insert<T: Into<TerminalRef>>(&mut self, key: NonTerminalRef, terminal: T) -> bool {
        self.sets
            .get_mut(&key)
            .map(|terminal_set| terminal_set.insert(terminal.into()))
            .unwrap_or(false)
    }

    /// Sets the terminals for `lhs` to the union of `lhs` and `rhs`.
    fn union_of_sets(&mut self, lhs: NonTerminalRef, rhs: &NonTerminalRef) -> bool {
        let mut changed = false;

        // get all terminals from the rhs non-terminal
        let first_terminal_from_rhs_non_terminal = self.sets.get(rhs).cloned().unwrap_or_default();
        self.sets.entry(lhs).and_modify(|terminal_set| {
            for terminal in first_terminal_from_rhs_non_terminal {
                changed = terminal_set.insert(terminal);
            }
        });

        changed
    }

    #[allow(unused)]
    fn human_readable_format(&self, grammar_table: &GrammarTable) -> String {
        let terminals = grammar_table.terminals().collect::<Vec<_>>();
        let nonterminals = grammar_table.non_terminals().collect::<Vec<_>>();

        let lines = self
            .sets
            .iter()
            .map(|(nonterm, terms)| {
                let rhs = terms
                    .iter()
                    .map(|term| terminals[term.as_usize()].to_string())
                    .collect::<Vec<_>>();

                format!(
                    "{}: {}",
                    nonterminals[nonterm.as_usize()].as_ref(),
                    rhs.join(", ")
                )
            })
            .collect::<Vec<_>>();

        lines.join("\n")
    }
}

impl AsRef<HashMap<NonTerminalRef, HashSet<TerminalRef>>> for SymbolRefSet {
    fn as_ref(&self) -> &HashMap<NonTerminalRef, HashSet<TerminalRef>> {
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
    let eof_terminal_ref = grammar_table.eof_terminal_ref();

    grammar_table
        .productions()
        .take(1)
        .map(|first_production| ItemRef::new(first_production, 0, eof_terminal_ref))
        .collect::<ItemSet>()
}

fn build_first_set_ref<'a>(
    grammar_table: &'a GrammarTable,
    nullable_nonterminals: &HashSet<NonTerminal<'a>>,
) -> SymbolRefSet {
    let nullable_nonterminal_refs = nullable_nonterminals
        .iter()
        .filter_map(|nt| grammar_table.non_terminal_mapping(nt))
        .collect::<Vec<_>>();

    let non_terminals = grammar_table
        .non_terminals()
        .filter_map(|nt| grammar_table.non_terminal_mapping(&nt))
        .collect::<Vec<_>>();
    let mut first_set = SymbolRefSet::new(&non_terminals);

    // builtin guarantee
    let epsilon_ref = grammar_table.terminal_mapping(&BuiltinTerminals::Epsilon.into());

    // map nullable nonterminals to epsilon
    if let Some(epsilon_ref) = epsilon_ref {
        for nullable_nonterm_ref in nullable_nonterminal_refs {
            first_set.insert(nullable_nonterm_ref, epsilon_ref);
        }
    // there should never be a case where there are nonterminal refs and no epison symbol.
    } else if !nullable_nonterminal_refs.is_empty() {
        unreachable!()
    }

    // set the initial terminal for each production
    let initial_terminals_of_productions =
        grammar_table.productions().filter_map(|production_ref| {
            let lhs_idx = production_ref.lhs;
            let lhs_non_terminal = non_terminals[lhs_idx.as_usize()];

            if let Some(SymbolRef::Terminal(first_terminal)) = production_ref.rhs.get(0) {
                // if the the first terminal in the pattern isn't epsilon, add it.
                Some((lhs_non_terminal, first_terminal))
            } else {
                None
            }
        });

    // map initial terminals in each proudction to their non-terminal
    for (non_terminal, first_terminal) in initial_terminals_of_productions {
        first_set.insert(non_terminal, *first_terminal);
    }

    let mut changed = true;

    while changed {
        changed = false;
        // set the initial terminal for each production
        for production_ref in grammar_table.productions() {
            let lhs_idx = production_ref.lhs;
            let lhs_non_terminal = non_terminals[lhs_idx.as_usize()];

            if let Some(SymbolRef::NonTerminal(idx)) = production_ref.rhs.get(0) {
                // get all terminals from the first non_terminal
                let first_rhs_non_terminal = non_terminals[idx.as_usize()];
                if first_set.union_of_sets(lhs_non_terminal, &first_rhs_non_terminal) {
                    changed = true;
                }
            }
        }
    }

    first_set
}

fn find_nullable_nonterminals(grammar_table: &GrammarTable) -> HashSet<NonTerminal> {
    let non_terminals = grammar_table.non_terminals().collect::<Vec<_>>();
    let terminals = grammar_table.terminals().collect::<Vec<_>>();
    let mut nullable_nonterminal_productions = HashSet::new();

    let mut done = false;
    while !done {
        // assume done unless a change happens.
        done = true;
        for production in grammar_table.productions() {
            let lhs_id = production.lhs;
            let lhs = non_terminals[lhs_id.as_usize()];

            // validate that the production isn't already nullable
            if !nullable_nonterminal_productions.contains(&lhs) {
                let first_rhs_is_terminal = production.rhs.get(0).and_then(|sotr| match sotr {
                    SymbolRef::NonTerminal(_) => None,
                    SymbolRef::Terminal(idx) => terminals.get(idx.as_usize()),
                });
                if first_rhs_is_terminal
                    == Some(&Terminal::new(BuiltinTerminals::Epsilon.as_terminal()))
                {
                    nullable_nonterminal_productions.insert(lhs);
                    done = false
                } else {
                    // check that the production doesn't contain a terminal or is not nullable.
                    let all_nullable = production.rhs.iter().any(|sotr| match sotr {
                        SymbolRef::NonTerminal(idx) => {
                            let non_terminal = non_terminals.get(idx.as_usize()).unwrap();
                            nullable_nonterminal_productions.contains(non_terminal)
                        }
                        SymbolRef::Terminal(_) => false,
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

    fn beta(&self) -> &[SymbolRef] {
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
    items: ordered_set::OrderedSet<ItemRef<'a>>,
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
        self.items
            .as_ref()
            .iter()
            .map(|item_ref| {
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

fn first(first_symbol_sets: &SymbolRefSet, beta_sets: &[&[SymbolRef]]) -> Vec<TerminalRef> {
    let mut in_firsts = HashSet::new();
    let mut firsts = vec![];

    for set in beta_sets {
        let first_symbol_in_beta = set.first();

        match first_symbol_in_beta {
            Some(SymbolRef::Terminal(term_ref)) => {
                let first_added_to_set = in_firsts.insert(*term_ref);
                if first_added_to_set {
                    firsts.push(*term_ref)
                }
            }
            Some(SymbolRef::NonTerminal(nt_ref)) => {
                if let Some(nt_firsts) = first_symbol_sets.sets.get(nt_ref) {
                    for &term_ref in nt_firsts {
                        let first_added_to_set = in_firsts.insert(term_ref);
                        if first_added_to_set {
                            firsts.push(term_ref)
                        }
                    }
                };
            }
            None => {}
        };
    }

    firsts
}

fn follow(first_symbol_sets: &SymbolRefSet, beta_sets: &[&[SymbolRef]]) -> Vec<TerminalRef> {
    for set in beta_sets {
        let firsts = first(first_symbol_sets, &[set]);
        // break the loop if the first set returns.
        if !firsts.is_empty() {
            return firsts;
        }
    }

    vec![]
}

/// Generates the closure of a `ItemSet` using the following algorithm.
///
/// ```ignore
/// Closure(I)
/// repeat
///     for (each item [ A -> ?.B?, a ] in I )
///         for (each production B -> ? in G’)
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

    let nullable_nonterms = find_nullable_nonterminals(grammar_table);
    let first_symbolref_set = build_first_set_ref(grammar_table, &nullable_nonterms);

    let mut set = i.items;

    let mut changed = true;
    while changed {
        changed = false;

        for item in set.clone() {
            let lookahead = item.lookahead;
            let beta = item.beta();
            let symbol_after_dot_position = item.symbol_after_dot();

            // handles for terminals and end of production.
            let maybe_next_symbol_after_dot_is_non_terminal = match symbol_after_dot_position {
                Some(SymbolRef::NonTerminal(s)) => Some(*s),
                _ => None,
            };

            if let Some(non_terminal_ref) = maybe_next_symbol_after_dot_is_non_terminal {
                let follow_set = {
                    let lookahead_set = [SymbolRef::Terminal(lookahead)];
                    follow(&first_symbolref_set, &[beta, &lookahead_set])
                        .into_iter()
                        .collect::<Vec<_>>()
                };

                let matching_productions = grammar_table
                    .productions()
                    .filter(|production| production.lhs == non_terminal_ref);

                for production in matching_productions {
                    let new_item = follow_set
                        .iter()
                        .map(|lookahead| ItemRef::new(production, 0, *lookahead));

                    for new in new_item {
                        let new_item_inserted = set.insert(new.clone());
                        if new_item_inserted {
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    let new_set = set.into_iter().collect::<Vec<_>>();

    ItemSet::new(new_set)
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
    // reverse the initial set so it can be popped.
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
    item_sets: ordered_set::OrderedSet<ItemSet<'a>>,
}

impl<'a> ItemCollection<'a> {
    fn states(&self) -> usize {
        self.item_sets.len()
    }

    /// Returns a boolean signifying a value is already in the set.
    fn contains(&self, new_set: &ItemSet<'a>) -> bool {
        self.item_sets.contains(new_set)
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
fn build_canonical_collection(grammar_table: &GrammarTable) -> ItemCollection {
    use ordered_set::OrderedSet;

    let mut collection = ItemCollection::default();

    let initial_item_set = initial_item_set(grammar_table);
    let s0 = closure(grammar_table, initial_item_set);

    let mut changing = collection.insert(s0);

    let mut new_states = OrderedSet::default();
    let mut new_sets_idx = 0_usize;

    while changing {
        changing = false;

        for parent_state in collection.item_sets.as_ref()[new_sets_idx..].iter() {
            let symbols_after_dot = {
                let mut symbol_after_dot = parent_state
                    .items
                    .as_ref()
                    .iter()
                    .filter_map(|item| item.symbol_after_dot().copied())
                    .collect::<Vec<_>>();
                symbol_after_dot.dedup();

                symbol_after_dot.into_iter()
            };

            for symbol_after_dot in symbols_after_dot {
                let new_state = goto(grammar_table, parent_state, symbol_after_dot);

                if !collection.contains(&new_state) {
                    new_states.insert(new_state.clone());
                }
            }
        }

        // bump the offset to account for only new states.
        new_sets_idx = collection.states();

        // insert all new states
        for new_state in new_states {
            // if there are new states to insert, mark the collection as
            // changing.
            changing = collection.insert(new_state);
        }
        // reset the new states
        new_states = OrderedSet::default();
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
            let lookahead_terminal_ref = &i.lookahead;
            let lookahead_terminal = terminals
                .get(lookahead_terminal_ref.as_usize())
                .ok_or_else(|| {
                    TableGenError::new(TableGenErrorKind::UnknownTerminal)
                        .with_data(format!("{}", &i.lookahead))
                })?;

            let is_goal = Some(i.production.lhs)
                == grammar_table
                    .non_terminal_mapping(&NonTerminal::from(BuiltinNonTerminals::Goal));
            let is_goal_acceptor =
                is_goal && symbol_after_dot.is_none() && (*lookahead_terminal == eof_terminal);

            // shift is symbol is both a terminal and not the end of input.
            if let Some(SymbolRef::Terminal(a)) = symbol_after_dot {
                let sk = goto(grammar_table, sx, SymbolRef::Terminal(a));
                let k = canonical_collection.id_from_set(&sk);

                if let Some(k) = k {
                    action_table[a.as_usize()][x] = Action::Shift(StateId::unchecked_new(k));
                    continue;
                }
            }

            // if it's the start action, accept
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
        for n in nt {
            let sk = goto(grammar_table, sx, SymbolRef::NonTerminal(n));

            // find the first set that contains all elements of sk.
            let k = canonical_collection
                .item_sets
                .as_ref()
                .iter()
                .position(|sx| sk.items.as_ref().starts_with(&sx.items.as_ref()));

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

    const TEST_GRAMMAR: &str = "
<E> ::= <T> - <E>
<E> ::= <T>
<T> ::= <F> * <T>
<T> ::= <F>
<F> ::= n
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

        assert_eq!(nullable_nonterminals, vec![NonTerminal::new("<factor>")])
    }

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

        let nullable_terms = find_nullable_nonterminals(&grammar_table);
        let first_sets = build_first_set_ref(&grammar_table, &nullable_terms);

        let mut got = first_sets
            .sets
            .into_iter()
            .map(|(nt, terms)| {
                let terms = terms
                    .into_iter()
                    .filter_map(|t| terminals.get(t.as_usize()).cloned())
                    .collect::<HashSet<_>>();

                (nonterminals[nt.as_usize()].clone(), terms)
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
            .zip(expected_productions_per_state.into_iter());
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
            let grammar_table = load_grammar(&self.grammar_table).unwrap();
            let nullable_terms = find_nullable_nonterminals(&grammar_table);
            let first_sets = build_first_set_ref(&grammar_table, &nullable_terms);

            for (&nt, &expected_terms) in self.expected_first_set_pairings.iter() {
                let key = grammar_table
                    .non_terminal_mapping(&NonTerminal::from(nt))
                    .unwrap();
                let first_set = first_sets.sets.get(&key).unwrap();
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
    #[ignore = "broken"]
    fn should_correctly_generate_closure_sets_for_recursive_grammar_two() {
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
                <UnaryOperator> ::= <Primary>
                <Postfix> ::= <Postfix> Token::LeftBracket <Expression> Token::RightBracket
                <Postfix> ::= <Postfix> Token::LeftParen Token::RightParen
                <Postfix> ::= <Postfix> Token::LeftParen <ArgumentExpressionList> Token::RightParen
                <Postfix> ::= <Postfix> Token::Dot Token::Identifier 
                <Postfix> ::= <Postfix> Token::Arrow Token::Identifier
                <Postfix> ::= <Postfix> Token::PlusPlus
                <Postfix> ::= <Postfix> Token::MinusMinus
                <ArgumentExpressionList> ::= <Assigment>
                <Primary> ::= Token::Identifier  
                <Primary> ::= <Constant>
                <Primary> ::= Token::StringLiteral
                <Primary> ::= Token::LeftParen <Expression> Token::RightParen
                <Constant> ::= Token::IntegerConstant
                <Constant> ::= Token::CharacterConstant
                <Constant> ::= Token::FloatingConstant
                ",
            )
            .with_expected_state_production_assertion(0, 166)
            //.with_expected_states_cnt(140)
            .test();
    }
}
