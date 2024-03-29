use std::collections::{HashMap, HashSet};

/// A predefined set of reserved non-terminals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinNonTerminals {
    Goal,
}

impl BuiltinNonTerminals {
    pub(crate) fn as_non_terminals(&self) -> &'static str {
        match self {
            Self::Goal => "<*>",
        }
    }
}

/// A predefined set of reserved terminals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinTerminals {
    Epsilon,
    Eof,
    EndL,
}

impl BuiltinTerminals {
    /// Returns a boolean signifying if a given string representation of a
    /// terminal matches one of the builtin terminals.
    pub fn is_builtin<S: AsRef<str>>(terminal: S) -> bool {
        let val = terminal.as_ref();
        [Self::Epsilon, Self::Eof, Self::EndL]
            .iter()
            .map(|builtin| builtin.as_terminal())
            .any(|builtin| builtin == val)
    }

    pub(crate) fn as_terminal(&self) -> &'static str {
        match self {
            BuiltinTerminals::Epsilon => "<epsilon>",
            BuiltinTerminals::Eof => "<$>",
            BuiltinTerminals::EndL => "<endl>",
        }
    }
}

/// An enum for storing a symbol within a given grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Symbol<'a> {
    NonTerminal(NonTerminal<'a>),
    Terminal(Terminal<'a>),
}

impl<'a> std::fmt::Display for Symbol<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Symbol::NonTerminal(s) => s.fmt(f),
            Symbol::Terminal(t) => t.fmt(f),
        }
    }
}

/// A wrapper type for non-terminals that reference the grammar table.
#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NonTerminalRef(usize);

impl NonTerminalRef {
    pub(crate) fn new(non_terminal: usize) -> Self {
        Self(non_terminal)
    }

    pub(crate) fn as_usize(&self) -> usize {
        self.0
    }
}

impl From<usize> for NonTerminalRef {
    fn from(val: usize) -> Self {
        Self::new(val)
    }
}

impl std::fmt::Display for NonTerminalRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Adds 1 as production is 1 indexed in human-readable format but 0
        // indexed internally. This is probably going to haunt me at some
        // point.
        write!(f, "S{}", &self.0 + 1)
    }
}

/// A wrapper type for terminals that reference the grammar table.
#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TerminalRef(usize);

impl TerminalRef {
    pub(crate) fn new(terminal_id: usize) -> Self {
        Self(terminal_id)
    }

    pub(crate) fn as_usize(&self) -> usize {
        self.0
    }
}

impl From<usize> for TerminalRef {
    fn from(val: usize) -> Self {
        Self::new(val)
    }
}

impl std::fmt::Display for TerminalRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Adds 1 as production is 1 indexed in human-readable format but 0
        // indexed internally. This is probably going to haunt me at some
        // point.
        write!(f, "T{}", &self.0 + 1)
    }
}

/// An enum for storing a symbol reference id within a given grammar.
#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SymbolRef {
    NonTerminal(NonTerminalRef),
    Terminal(TerminalRef),
}

impl std::fmt::Display for SymbolRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymbolRef::NonTerminal(id) => write!(f, "{}", id),
            SymbolRef::Terminal(id) => write!(f, "{}", id),
        }
    }
}

/// A structure representing a grammar production.
#[derive(Debug, Hash, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProductionRef {
    /// The left-hand side symbol of a production
    pub lhs: NonTerminalRef,
    /// One or more right-hand side symbols of a production.
    pub rhs: Vec<SymbolRef>,
}

impl ProductionRef {
    pub(crate) fn new(lhs: NonTerminalRef, rhs: Vec<SymbolRef>) -> Option<Self> {
        let production = Self::new_unchecked(lhs, rhs);

        if production.is_valid() {
            Some(production)
        } else {
            None
        }
    }

    fn new_unchecked(lhs: NonTerminalRef, rhs: Vec<SymbolRef>) -> Self {
        Self { lhs, rhs }
    }

    fn is_valid(&self) -> bool {
        let production = self
            .rhs
            .iter()
            .all(|items| items == &SymbolRef::NonTerminal(self.lhs));

        !(self.rhs.is_empty() || production)
    }

    pub fn rhs_len(&self) -> usize {
        self.rhs.len()
    }
}

impl std::fmt::Display for ProductionRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lhs = format!("S{}", self.lhs.as_usize() + 1);
        let rhs = self
            .rhs
            .iter()
            .map(|sot| sot.to_string())
            .collect::<Vec<_>>()
            .join(" ");

        write!(f, "{} ::= {}", lhs, rhs)
    }
}

/// A wrapper type for non-terminals borrowed from the grammar table.
#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub struct NonTerminal<'a>(&'a str);

impl<'a> NonTerminal<'a> {
    pub(crate) fn new(non_terminal: &'a str) -> Self {
        Self(non_terminal)
    }
}
impl<'a> AsRef<str> for NonTerminal<'a> {
    fn as_ref(&self) -> &str {
        self.0
    }
}

impl<'a> std::fmt::Display for NonTerminal<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.0)
    }
}

impl<'a> From<BuiltinNonTerminals> for NonTerminal<'a> {
    fn from(val: BuiltinNonTerminals) -> Self {
        Self::new(val.as_non_terminals())
    }
}

impl<'a> From<&'a str> for NonTerminal<'a> {
    fn from(val: &'a str) -> Self {
        Self::new(val)
    }
}

/// A wrapper type for terminals borrowed from the grammar table.
#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Terminal<'a>(&'a str);

impl<'a> Terminal<'a> {
    pub(crate) fn new(terminal: &'a str) -> Self {
        Self(terminal)
    }
}

impl<'a> AsRef<str> for Terminal<'a> {
    fn as_ref(&self) -> &str {
        self.0
    }
}

impl<'a> From<&'a str> for Terminal<'a> {
    fn from(val: &'a str) -> Self {
        Terminal::new(val)
    }
}

impl<'a> From<BuiltinTerminals> for Terminal<'a> {
    fn from(val: BuiltinTerminals) -> Self {
        Terminal::new(val.as_terminal())
    }
}

impl<'a> std::fmt::Display for Terminal<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.0)
    }
}

/// A mapping of non-terminal symbols to their corresponding first terminal symbols.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct FirstSymbolSet {
    sets: HashMap<NonTerminalRef, crate::ordered_set::OrderedSet<TerminalRef>>,
}

impl FirstSymbolSet {
    pub fn new<NT: AsRef<[NonTerminalRef]>>(non_terminals: NT) -> Self {
        let sets = non_terminals
            .as_ref()
            .iter()
            .fold(HashMap::new(), |mut acc, &non_terminal| {
                acc.insert(non_terminal, crate::ordered_set::OrderedSet::new());
                acc
            });
        Self { sets }
    }

    /// Inserts a terminal into a non-terminals's set returning true if it already exists.
    pub fn insert<T: Into<TerminalRef>>(&mut self, key: NonTerminalRef, terminal: T) -> bool {
        self.sets
            .get_mut(&key)
            .map(|terminal_set| terminal_set.insert(terminal.into()))
            .unwrap_or(false)
    }

    /// Sets the terminals for `lhs` to the union of `lhs` and `rhs`.
    pub fn union_of_sets(&mut self, lhs: NonTerminalRef, rhs: &NonTerminalRef) -> bool {
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

    /// Returns an immutable iterator over the [TerminalRef] mappings for each
    /// [NonTerminalRef].
    pub fn iter(
        &self,
    ) -> std::collections::hash_map::Iter<NonTerminalRef, crate::ordered_set::OrderedSet<TerminalRef>>
    {
        self.sets.iter()
    }

    /// Returns a human friendly representation of the set.
    pub fn human_readable_format(&self, grammar_table: &GrammarTable) -> String {
        let terminals = grammar_table.terminals().collect::<Vec<_>>();
        let nonterminals = grammar_table.non_terminals().collect::<Vec<_>>();

        let lines = self
            .sets
            .iter()
            .map(|(nonterm, terms)| {
                let rhs = terms
                    .as_ref()
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

impl AsRef<HashMap<NonTerminalRef, crate::ordered_set::OrderedSet<TerminalRef>>>
    for FirstSymbolSet
{
    fn as_ref(&self) -> &HashMap<NonTerminalRef, crate::ordered_set::OrderedSet<TerminalRef>> {
        &self.sets
    }
}

impl IntoIterator for FirstSymbolSet {
    type Item = (NonTerminalRef, crate::ordered_set::OrderedSet<TerminalRef>);

    type IntoIter = std::collections::hash_map::IntoIter<
        NonTerminalRef,
        crate::ordered_set::OrderedSet<TerminalRef>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.sets.into_iter()
    }
}

/// A representation of all symbols, and productions of a given grammar. This
/// type functions as a source of truth for both human readable rendering and
/// state machine generation.
#[derive(Debug, Default, PartialEq)]
pub struct GrammarTable {
    non_terminals: HashMap<String, usize>,
    terminals: HashMap<String, usize>,
    productions: Vec<ProductionRef>,

    first_sets: FirstSymbolSet,

    eof_terminal_ref: Option<TerminalRef>,
}

impl GrammarTable {
    pub(crate) const ROOT_PRODUCTION_IDX: usize = 0;

    /// Adds a non-terminal to the table, returning its index. If the
    /// non-terminal already exists, the index to the previously added
    /// non-terminal is returned.
    fn add_non_terminal_mut<NT: AsRef<str>>(&mut self, non_terminal: NT) -> usize {
        let non_terminal = non_terminal.as_ref();
        let new_id = self.non_terminals.len();

        self.non_terminals
            .entry(non_terminal.to_string())
            .or_insert(new_id);

        // safe to unwrap due to above guarantee
        self.non_terminals.get(non_terminal).copied().unwrap()
    }

    /// Adds a terminal to the table, returning its index. If the terminal already
    /// exists, the index to the previously added terminal is returned.
    fn add_terminal_mut<S: AsRef<str>>(&mut self, terminal: S) -> usize {
        let terminal = terminal.as_ref();
        let new_id = self.terminals.len();

        self.terminals.entry(terminal.to_string()).or_insert(new_id);
        // safe to unwrap due to above guarantee
        self.terminals.get(terminal).copied().unwrap()
    }

    fn add_production_mut(&mut self, production: ProductionRef) {
        self.productions.push(production);
    }

    /// Allocates, if not defined, and assigns a given terminal representation
    /// as the End-of-file delimiter for a grammar.
    pub fn declare_eof_terminal<T: AsRef<str>>(
        &mut self,
        terminal: T,
    ) -> Result<usize, GrammarLoadError> {
        let repr = terminal.as_ref();

        let terminal_id = self.add_terminal_mut(repr);
        self.eof_terminal_ref = Some(TerminalRef(terminal_id));

        Ok(terminal_id)
    }

    /// Returns a reference to the end-of-file terminal for a grammar.
    pub fn eof_terminal_ref(&self) -> TerminalRef {
        match self.eof_terminal_ref {
            Some(eof_tok) => eof_tok,
            None => self.builtin_terminal_mapping(&BuiltinTerminals::Eof),
        }
    }

    /// Returns an iterator over all non-terminal symbols in a given grammar.
    pub fn non_terminals(&self) -> NonTerminalIterator {
        NonTerminalIterator::new(self)
    }

    /// Returns an iterator over all terminal symbols in a given grammar.
    pub fn terminals(&self) -> TerminalIterator {
        TerminalIterator::new(self)
    }

    pub(crate) fn non_terminal_mapping(
        &self,
        non_terminal: &NonTerminal,
    ) -> Option<NonTerminalRef> {
        self.non_terminals
            .get(non_terminal.0)
            .map(|id| NonTerminalRef(*id))
    }

    pub(crate) fn terminal_mapping(&self, terminal: &Terminal) -> Option<TerminalRef> {
        self.terminals.get(terminal.0).map(|id| TerminalRef(*id))
    }

    /// Similarly to terminal_mapping, this looks up a [TerminalRef], however
    /// the builtin guaranteed removes the need to return an [Option].
    pub(crate) fn builtin_terminal_mapping(&self, terminal: &BuiltinTerminals) -> TerminalRef {
        self.terminals
            .get(terminal.as_terminal())
            .map(|id| TerminalRef(*id))
            .unwrap()
    }

    /// Returns an iterator over all productions in a given grammar.
    pub fn productions(&self) -> impl Iterator<Item = &ProductionRef> {
        self.productions.iter()
    }

    /// generates the first set from the grammar, finalizing it.
    pub fn finalize(&mut self) {
        let non_terminals = self.non_terminals().collect::<Vec<_>>();
        let terminals = self.terminals().collect::<Vec<_>>();

        let nullable_terms =
            find_nullable_nonterminals(&self.productions, &non_terminals, &terminals);
        let first_sets = build_first_set_ref(self, &nullable_terms);
        self.first_sets = first_sets;
    }

    pub fn first_set(&self) -> &FirstSymbolSet {
        &self.first_sets
    }
}

impl std::fmt::Display for GrammarTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let header = "Grammar Table
-------------";

        let non_terminals = self
            .non_terminals()
            .enumerate()
            .map(|(idx, non_terminal)| (idx + 1, non_terminal))
            .fold(String::new(), |mut acc, (rule_id, non_terminal)| {
                let repr = format!("{}. '{}'\n", rule_id, non_terminal);
                acc.push_str(&repr);
                acc
            });
        let terminals = self
            .terminals()
            .enumerate()
            .map(|(idx, non_terminal)| (idx + 1, non_terminal))
            .fold(String::new(), |mut acc, (rule_id, terminal)| {
                let repr = format!("{}. '{}'\n", rule_id, terminal);
                acc.push_str(&repr);
                acc
            });

        let productions = self
            .productions
            .iter()
            .enumerate()
            // 1-indexed
            .map(|(idx, production)| (idx + 1, production))
            .fold(String::new(), |mut acc, (rule_id, production)| {
                let repr = format!("{}. '{}'\n", rule_id, production);
                acc.push_str(&repr);
                acc
            });
        write!(
            f,
            "{}\nNON-TERMINALS\n{}\nTERMINALS\n{}\nPRODUCTIONS\n{}",
            header, non_terminals, terminals, productions
        )
    }
}

/// Provides methods for initializing a [GrammarTable] with expected terms.
pub trait GrammarInitializer {
    fn initialize_table() -> GrammarTable;
}

/// Defines a [GrammarTable] builder that includes a Goal non-terminal.
pub struct DefaultInitializedGrammarTableSansBuiltins;

impl DefaultInitializedGrammarTableSansBuiltins {
    const DEFAULT_NON_TERMINALS: [BuiltinNonTerminals; 1] = [BuiltinNonTerminals::Goal];
}

impl GrammarInitializer for DefaultInitializedGrammarTableSansBuiltins {
    fn initialize_table() -> GrammarTable {
        let mut grammar_table = GrammarTable::default();

        for builtin_non_terminals in Self::DEFAULT_NON_TERMINALS {
            grammar_table.add_non_terminal_mut(builtin_non_terminals.as_non_terminals());
        }

        grammar_table
    }
}

/// Defines a [GrammarTable] builder that includes a Goal non-terminal and common
/// terminals.
pub struct DefaultInitializedGrammarTableBuilder;

impl DefaultInitializedGrammarTableBuilder {
    const DEFAULT_NON_TERMINALS: [BuiltinNonTerminals; 1] = [BuiltinNonTerminals::Goal];
    const DEFAULT_TERMINALS: [BuiltinTerminals; 3] = [
        BuiltinTerminals::Epsilon,
        BuiltinTerminals::Eof,
        BuiltinTerminals::EndL,
    ];
}

impl GrammarInitializer for DefaultInitializedGrammarTableBuilder {
    fn initialize_table() -> GrammarTable {
        let mut grammar_table = GrammarTable::default();

        for builtin_non_terminals in Self::DEFAULT_NON_TERMINALS {
            grammar_table.add_non_terminal_mut(builtin_non_terminals.as_non_terminals());
        }

        // add default terminals
        for builtin_terminals in Self::DEFAULT_TERMINALS {
            let non_terminal_string_repr = builtin_terminals.as_terminal().to_string();
            grammar_table.add_terminal_mut(non_terminal_string_repr);
        }

        grammar_table
    }
}

/// Defines a [GrammarTable] builder that includes a Goal non-terminals, Goal
/// production and common terminals.
pub struct DefaultInitializedWithGoalProductionGrammarTableBuilder;

impl GrammarInitializer for DefaultInitializedWithGoalProductionGrammarTableBuilder {
    fn initialize_table() -> GrammarTable {
        let mut grammar_table = DefaultInitializedGrammarTableBuilder::initialize_table();

        // initial table
        let root_production_idx = NonTerminalRef::new(GrammarTable::ROOT_PRODUCTION_IDX);
        let root_production = ProductionRef::new_unchecked(root_production_idx, vec![]);
        grammar_table.add_production_mut(root_production);

        grammar_table
    }
}

/// An ordered iterator over all non_terminals in a grammar table.
pub struct NonTerminalIterator<'a> {
    non_terminals: Vec<&'a str>,
}

impl<'a> NonTerminalIterator<'a> {
    fn new(grammar_table: &'a GrammarTable) -> Self {
        let mut values = grammar_table
            .non_terminals
            .iter()
            .map(|(key, value)| (key.as_str(), value))
            .collect::<Vec<_>>();
        // reverse the order so first production pops off the top of the stack.
        values.sort_by(|(_, a), (_, b)| b.cmp(a));

        Self {
            non_terminals: values.into_iter().map(|(key, _)| key).collect(),
        }
    }
}

impl<'a> Iterator for NonTerminalIterator<'a> {
    type Item = NonTerminal<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.non_terminals.pop().map(NonTerminal)
    }
}

/// An ordered iterator over all terminals in a grammar table.
pub struct TerminalIterator<'a> {
    terminals: Vec<&'a str>,
}

impl<'a> TerminalIterator<'a> {
    fn new(grammar_table: &'a GrammarTable) -> Self {
        let mut values = grammar_table
            .terminals
            .iter()
            .map(|(key, value)| (key.as_str(), value))
            .collect::<Vec<_>>();
        // reverse the order so first production pops off the top of the stack.
        values.sort_by(|(_, a), (_, b)| b.cmp(a));

        Self {
            terminals: values.into_iter().map(|(key, _)| key).collect(),
        }
    }
}

impl<'a> Iterator for TerminalIterator<'a> {
    type Item = Terminal<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.terminals.pop().map(Terminal)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum GrammarLoadErrorKind {
    /// Represents a grammar without a terminating (goal) production.
    NoTerminalProduction,
    /// An invalid production was defined.
    InvalidProduction,
    /// Represents any error arising from two conflicting, often identical,
    /// productions being defined on a grammar.
    ConflictingProduction,
    /// Any error stemming from an invalid or unknown terminal symbol
    /// representation.
    InvalidTerminal,
}

impl std::fmt::Display for GrammarLoadErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoTerminalProduction => {
                write!(f, "grammar does not include a terminal production")
            }
            Self::InvalidProduction => write!(f, "provided production is invalid",),
            Self::ConflictingProduction => {
                write!(f, "provided production conflicts with existing production",)
            }
            Self::InvalidTerminal => write!(f, "provided terminal is not registered with grammer"),
        }
    }
}

/// An error type representing any error arising while defining a grammar.
#[derive(Debug, PartialEq, Eq)]
pub struct GrammarLoadError {
    kind: GrammarLoadErrorKind,
    data: Option<String>,
}

impl GrammarLoadError {
    pub(crate) fn new(kind: GrammarLoadErrorKind) -> Self {
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

impl std::fmt::Display for GrammarLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.data {
            Some(ctx) => write!(f, "{}: {}", &self.kind, ctx),
            None => write!(f, "{}", &self.kind),
        }
    }
}

/// A function for defining productions on a grammar individually.
pub fn define_production_mut<S: AsRef<str>>(
    grammar_table: &mut GrammarTable,
    line: S,
) -> Result<(), GrammarLoadError> {
    let trimmed_line = line.as_ref().trim();

    // validate the start of the line is a non-terminal
    if !trimmed_line.starts_with('<') {
        return Err(
            GrammarLoadError::new(GrammarLoadErrorKind::InvalidProduction)
                .with_data("doesn't start with a non-terminal"),
        );
    }

    // split the line at the assignment delimiter
    let mut split_line = trimmed_line.split("::=").collect::<Vec<_>>();
    if split_line.len() != 2 {
        return Err(
            GrammarLoadError::new(GrammarLoadErrorKind::InvalidProduction)
                .with_data("does not contain right-hand side"),
        );
    }

    // safe to assume this will have a value from above checks.
    let rhs = split_line
        .pop()
        .map(|rhs| rhs.trim())
        // break each rhs predicate up by whitespace
        .map(|str| str.split_whitespace())
        .unwrap();
    let lhs = split_line.pop().map(|lhs| lhs.trim()).unwrap();

    // retrieve the LHS non-terminal.
    let lhs_non_terminal = non_terminal_value_from_str(lhs).ok_or_else(|| {
        GrammarLoadError::new(GrammarLoadErrorKind::InvalidProduction)
            .with_data("doesn't start with non-terminal")
    })?;

    let production_id = grammar_table.add_non_terminal_mut(lhs_non_terminal);
    let production_ref = NonTerminalRef::from(production_id);
    let mut production = ProductionRef::new_unchecked(production_ref, vec![]);

    // add terminals and fill the production.
    for elem in rhs {
        if let Some(non_terminal) = non_terminal_value_from_str(elem) {
            let symbol_id = {
                let non_terminal_id = grammar_table.add_non_terminal_mut(non_terminal);
                let non_terminal_ref = NonTerminalRef::new(non_terminal_id);

                SymbolRef::NonTerminal(non_terminal_ref)
            };
            production.rhs.push(symbol_id);
        // validate all terminal values are single length.
        } else if let Some(terminal) = terminal_value_from_str(elem) {
            let terminal_id = {
                let terminal_id = grammar_table.add_terminal_mut(terminal);
                let terminal_ref = TerminalRef::from(terminal_id);

                SymbolRef::Terminal(terminal_ref)
            };
            production.rhs.push(terminal_id);
        } else {
            return Err(
                GrammarLoadError::new(GrammarLoadErrorKind::InvalidProduction)
                    .with_data(format!("invalid rhs value ({})", elem)),
            );
        }
    }

    if grammar_table.productions.contains(&production) {
        Err(
            GrammarLoadError::new(GrammarLoadErrorKind::ConflictingProduction)
                .with_data(format!("{} ", &production)),
        )
    } else if !production.is_valid() {
        Err(
            GrammarLoadError::new(GrammarLoadErrorKind::InvalidProduction)
                .with_data(format!("{} ", &production)),
        )
    } else {
        grammar_table.add_production_mut(production);
        Ok(())
    }
}

fn find_nullable_nonterminals<'a>(
    productions: &'a [ProductionRef],
    non_terminals: &'a [NonTerminal],
    terminals: &'a [Terminal],
) -> HashSet<NonTerminal<'a>> {
    let mut nullable_nonterminal_productions = HashSet::new();

    let mut done = false;
    while !done {
        // assume done unless a change happens.
        done = true;
        for production in productions {
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

fn build_first_set_ref<'a>(
    grammar_table: &'a GrammarTable,
    nullable_nonterminals: &HashSet<NonTerminal<'a>>,
) -> FirstSymbolSet {
    let nullable_nonterminal_refs = nullable_nonterminals
        .iter()
        .filter_map(|nt| grammar_table.non_terminal_mapping(nt))
        .collect::<Vec<_>>();

    let non_terminals = grammar_table
        .non_terminals()
        .filter_map(|nt| grammar_table.non_terminal_mapping(&nt))
        .collect::<Vec<_>>();
    let mut first_set = FirstSymbolSet::new(&non_terminals);

    // builtin guarantee
    let epsilon_ref = grammar_table.terminal_mapping(&BuiltinTerminals::Epsilon.into());

    // map nullable nonterminals to epsilon
    if let Some(epsilon_ref) = epsilon_ref {
        for nullable_nonterm_ref in nullable_nonterminal_refs {
            first_set.insert(nullable_nonterm_ref, epsilon_ref);
        }
    // there should never be a case where there are nonterminal refs and no epsilon symbol.
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

/// A function for loading a single text representation of a grammar in one pass.
pub fn load_grammar<S: AsRef<str>>(input: S) -> Result<GrammarTable, GrammarLoadError> {
    let mut grammar_table =
        DefaultInitializedWithGoalProductionGrammarTableBuilder::initialize_table();

    // breakup input into enumerated lines.
    let lines = input
        .as_ref()
        .lines()
        .enumerate()
        .map(|(lineno, line)| (lineno + 1, line));

    let lines_containing_productions = lines
        // ignore commented lines.
        .filter(|(_, line)| !line.starts_with(';'))
        // ignore empty lines.
        .filter(|(_, line)| !line.chars().all(|c| c.is_whitespace()));

    for (lineno, line) in lines_containing_productions {
        define_production_mut(&mut grammar_table, line).map_err(|e| match e.data {
            Some(data) => {
                let line_annotated_data = format!("lineno {}: {}", lineno, data);
                GrammarLoadError::new(e.kind).with_data(line_annotated_data)
            }
            None => e,
        })?;
    }

    // add the first production to the goal.
    let root_production = grammar_table
        .productions()
        .next()
        .map(|production_ref| production_ref.lhs)
        .ok_or_else(|| GrammarLoadError::new(GrammarLoadErrorKind::NoTerminalProduction))?;
    let first_non_root_production = grammar_table
        .productions()
        .nth(1)
        .map(|production_ref| production_ref.lhs)
        .map(SymbolRef::NonTerminal)
        .ok_or_else(|| GrammarLoadError::new(GrammarLoadErrorKind::NoTerminalProduction))?;

    grammar_table.productions[root_production.as_usize()]
        .rhs
        .push(first_non_root_production);
    grammar_table.finalize();

    Ok(grammar_table)
}

/// Validates a non-terminal str representation conforms to conventions defined
/// within the grammar table. Returning a None value if the passed str doesn't
/// conform.
fn non_terminal_value_from_str(value: &str) -> Option<&str> {
    let trimmed_value = value.trim();

    let is_wrapped = trimmed_value.starts_with('<') && trimmed_value.ends_with('>');
    let is_not_empty = trimmed_value.len() > 2;
    let is_builtin = BuiltinTerminals::is_builtin(trimmed_value);

    // guarantee that it's a non_terminal and that it's not just an empty symbol `<>`
    if is_wrapped && is_not_empty && !is_builtin {
        Some(trimmed_value)
    } else {
        None
    }
}

/// Validates a terminal str representation conforms to conventions defined
/// within the grammar table. Returning a None value if the passed str doesn't
/// conform.
fn terminal_value_from_str(value: &str) -> Option<&str> {
    let trimmed_value = value.trim();

    Some(trimmed_value)
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
    fn should_parse_table_with_valid_test_grammar() {
        let grammar_table = load_grammar(TEST_GRAMMAR).unwrap();

        assert_eq!(2, grammar_table.non_terminals.len());
        // 3 builtins plus `(` and `)`
        assert_eq!(5, grammar_table.terminals.len());
        assert_eq!(7, grammar_table.productions.len());
    }

    #[test]
    fn should_error_on_conflicting_production() {
        let res = load_grammar(
            "
<expr> ::= ( <expr> )
<expr> ::= ( )
<expr> ::= ( ) 
        ",
        );

        assert_eq!(
            Err(GrammarLoadErrorKind::ConflictingProduction),
            res.map_err(|e| e.kind)
        );
    }

    #[test]
    fn should_iterate_non_termials_in_order() {
        let grammar = "
<expr> ::= ( <expr> )
<expr> ::= <addition>
<addition> ::= <expr> + <expr>  
        ";
        let grammar_table = load_grammar(grammar).unwrap();

        let mut non_terminal_iter = grammar_table.non_terminals();

        assert_eq!(
            Some(NonTerminal(BuiltinNonTerminals::Goal.as_non_terminals())),
            non_terminal_iter.next()
        );
        assert_eq!(Some(NonTerminal("<expr>")), non_terminal_iter.next());
        assert_eq!(Some(NonTerminal("<addition>")), non_terminal_iter.next());
        assert_eq!(None, non_terminal_iter.next());
    }

    #[test]
    fn should_iterate_terminals_in_order() {
        let grammar = "
<expr> ::= ( <expr> )
<expr> ::= <addition>
<addition> ::= <expr> + <expr>  
        ";
        let grammar_table = load_grammar(grammar).unwrap();

        let mut terminal_iter = grammar_table
            .terminals()
            // strip out the builtins for the sake of testing.
            .filter(|terminal| !terminal.0.starts_with('<'));

        assert_eq!(Some(Terminal("(")), terminal_iter.next());
        assert_eq!(Some(Terminal(")")), terminal_iter.next());
        assert_eq!(Some(Terminal("+")), terminal_iter.next());
        assert_eq!(None, terminal_iter.next());
    }
}
