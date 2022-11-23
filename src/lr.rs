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

fn initial_item_set<'a>(grammar_table: &'a GrammarTable) -> ItemSet<'a> {
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

#[derive(Hash, Debug, Clone, Copy, PartialEq, Eq)]
enum ItemSetParent {
    Root,
    Parent(usize),
}

impl Default for ItemSetParent {
    fn default() -> Self {
        Self::Root
    }
}

impl PartialOrd for ItemSetParent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;

        match (self, other) {
            (ItemSetParent::Root, ItemSetParent::Root) => Some(Ordering::Equal),
            (ItemSetParent::Root, ItemSetParent::Parent(_)) => Some(Ordering::Less),
            (ItemSetParent::Parent(_), ItemSetParent::Root) => Some(Ordering::Greater),
            (ItemSetParent::Parent(a), ItemSetParent::Parent(b)) => a.partial_cmp(b),
        }
    }
}

impl Ord for ItemSetParent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
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
    child_mapping: Vec<Vec<usize>>,
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
    fn insert(&mut self, parent: ItemSetParent, new_set: ItemSet<'a>) -> bool {
        let already_present = self.contains(&new_set);
        if !already_present {
            self.item_sets.push(new_set);
            self.child_mapping.push(vec![]);

            // add the new set to the parent's child mapping.
            let child_id = self.item_sets.len();
            match parent {
                ItemSetParent::Root => (),
                ItemSetParent::Parent(parent_id) => {
                    let _ = self
                        .child_mapping
                        .get_mut(parent_id)
                        .map(|parent_child_set| {
                            if !parent_child_set.contains(&child_id) {
                                parent_child_set.push(child_id);
                            }
                        });
                }
            };
        }

        // if it's not already present, then a value has been inserted.
        !already_present
    }

    fn into_ordered_iter(self) -> OrderedItemCollectionIter<'a> {
        let item_sets = self.item_sets.to_vec();

        OrderedItemCollectionIter(item_sets)
    }

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

    type IntoIter = OrderedItemCollectionIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_ordered_iter()
    }
}

/// Provides an ordered iterator over the `ItemSet`'s contained in a in a
/// canonical collection.
struct OrderedItemCollectionIter<'a>(Vec<ItemSet<'a>>);

impl<'a> Iterator for OrderedItemCollectionIter<'a> {
    type Item = ItemSet<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop()
    }
}

/// Constructs a canonical collection from a GrammarTable using the following algorithm.
///
/// ```ignore
/// s0 ← closure ( [S’→S,EOF] )
/// S ← {s0 }
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
    let mut collection = ItemCollection::default();

    let initial_item_set = initial_item_set(grammar_table);
    let s0 = closure(grammar_table, initial_item_set);

    let mut changing = collection.insert(ItemSetParent::Root, s0);
    let mut new_states = vec![];

    while changing {
        changing = false;

        for (parent_state_id, parent_state) in collection.item_sets.iter().enumerate() {
            let parent = ItemSetParent::Parent(parent_state_id);
            let parent_state_mapping = ItemSetParent::Parent(parent_state_id);
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
                let new_state = goto(grammar_table, &parent_state, symbol_after_dot);

                // Strips any items from the new state that exist in the parent
                // state.
                let non_duplicate_items = new_state
                    .items
                    .into_iter()
                    .filter(|item| !parent_state.contains(item));

                // Constructs the new state from the non duplicate item set, assigining a parent.
                let new_non_duplicate_state = non_duplicate_items.collect();

                if !collection.contains(&new_non_duplicate_state) {
                    new_states.push((parent_state_mapping, new_non_duplicate_state));
                }
            }
        }

        for (parent_state_mapping, new_state) in new_states {
            // if there are new states to insert, mark the collection as
            // changing.
            changing = collection.insert(parent_state_mapping, new_state);
        }
        new_states = vec![];
    }

    collection
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Action {
    Accept,
    Shift(usize),
    Reduce(usize),
    Invalid,
}

impl Default for Action {
    fn default() -> Self {
        Self::Invalid
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Goto {
    State(usize),
    Invalid,
}

impl Default for Goto {
    fn default() -> Self {
        Self::Invalid
    }
}

pub(crate) struct LrTable<'a> {
    goto: Vec<HashMap<Token<'a>, Goto>>,
    action: Vec<HashMap<Symbol<'a>, Action>>,
}

impl<'a> LrTable<'a> {
    pub(crate) fn new(
        goto: Vec<HashMap<Token<'a>, Goto>>,
        action: Vec<HashMap<Symbol<'a>, Action>>,
    ) -> Self {
        Self { goto, action }
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
) -> Result<LrTable<'a>, ParserGenError> {
    let mut goto: Vec<HashMap<Token<'a>, Goto>> = Vec::with_capacity(canonical_collection.states());
    let mut action: Vec<HashMap<Symbol<'a>, Action>> =
        Vec::with_capacity(canonical_collection.states());

    for sx in canonical_collection.clone().into_ordered_iter() {
        for i in &sx.items {}
    }

    todo!()
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
                Symbol::from(BuiltinSymbols::Goal),
                [Token::new("<integer>"), Token::new("+"), Token::new("(")]
                    .into_iter()
                    .collect::<HashSet<_>>(),
            ),
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
<F> ::= <identifier>";
        let grammar_table = load_grammar(grammar);

        assert!(grammar_table.is_ok());

        // safe to unwrap with assertion.
        let grammar_table = grammar_table.unwrap();

        let s0 = initial_item_set(&grammar_table);
        let s0 = closure(&grammar_table, s0);
        assert_eq!(s0.len(), 10);
        assert!(s0
            .human_readable_format(&grammar_table)
            .contains("<F> -> . <identifier> [*]"));
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
                grammar_table.ref_to_concrete(&symbol_after_dot).unwrap(),
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
<F> ::= <identifier>";
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
}
