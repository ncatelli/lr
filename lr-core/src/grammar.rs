use std::collections::hash_map::HashMap;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinTerminals {
    Epsilon,
    Eof,
    EndL,
}

impl BuiltinTerminals {
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
#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Hash, Clone, PartialEq, Eq)]
pub struct ProductionRef {
    pub lhs: NonTerminalRef,
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

#[derive(Debug, Default, PartialEq)]
pub struct GrammarTable {
    non_terminals: HashMap<String, usize>,
    terminals: HashMap<String, usize>,
    productions: Vec<ProductionRef>,

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

    pub fn declare_eof_terminal<T: AsRef<str>>(
        &mut self,
        terminal: T,
    ) -> Result<usize, GrammarLoadError> {
        let repr = terminal.as_ref();

        let terminal_id = self.add_terminal_mut(repr);
        self.eof_terminal_ref = Some(TerminalRef(terminal_id));

        Ok(terminal_id)
    }

    pub fn eof_terminal_ref(&self) -> TerminalRef {
        match self.eof_terminal_ref {
            Some(eof_tok) => eof_tok,
            None => self.builtin_terminal_mapping(&BuiltinTerminals::Eof),
        }
    }

    pub fn non_terminals(&self) -> NonTerminalIterator {
        NonTerminalIterator::new(self)
    }

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

    pub fn productions(&self) -> impl Iterator<Item = &ProductionRef> {
        self.productions.iter()
    }
}

impl std::fmt::Display for GrammarTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let header = "Grammar Table
-------------";

        let non_terminals = self
            .non_terminals()
            .enumerate()
            .map(|(id, non_terminal)| format!("{}. '{}'\n", id + 1, non_terminal))
            .collect::<String>();
        let terminals = self
            .terminals()
            .enumerate()
            .map(|(id, terminal)| format!("{}. '{}'\n", id + 1, terminal))
            .collect::<String>();

        let productions = self
            .productions
            .iter()
            .enumerate()
            // 1-indexed
            .map(|(idx, production)| (idx + 1, production))
            .map(|(idx, production)| format!("{}. {}\n", idx, production))
            .collect::<String>();

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
    NoTerminalProduction,
    InvalidProduction,
    ConflictingProduction,
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

#[derive(Debug, PartialEq, Eq)]
pub struct GrammarLoadError {
    kind: GrammarLoadErrorKind,
    data: Option<String>,
}

impl GrammarLoadError {
    pub fn new(kind: GrammarLoadErrorKind) -> Self {
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

impl std::fmt::Display for GrammarLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.data {
            Some(ctx) => write!(f, "{}: {}", &self.kind, ctx),
            None => write!(f, "{}", &self.kind),
        }
    }
}

pub fn define_production_mute<S: AsRef<str>>(
    grammar_table: &mut GrammarTable,
    line: S,
) -> Result<(), GrammarLoadError> {
    let trimmed_line = line.as_ref().trim();

    // validate the start of the line is a non-terminal
    if !trimmed_line.starts_with('<') {
        return Err(
            GrammarLoadError::new(GrammarLoadErrorKind::InvalidProduction)
                .with_data("doesn't start with a non-terminal".to_string()),
        );
    }

    // split the line at the assignment delimiter
    let mut split_line = trimmed_line.split("::=").collect::<Vec<_>>();
    if split_line.len() != 2 {
        return Err(
            GrammarLoadError::new(GrammarLoadErrorKind::InvalidProduction)
                .with_data("does not contain right-hand side".to_string()),
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
            .with_data("doesn't start with non-terminal".to_string())
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
        return Err(
            GrammarLoadError::new(GrammarLoadErrorKind::ConflictingProduction)
                .with_data(format!("{} ", &production)),
        );
    } else if !production.is_valid() {
        return Err(
            GrammarLoadError::new(GrammarLoadErrorKind::InvalidProduction)
                .with_data(format!("{} ", &production)),
        );
    } else {
        grammar_table.add_production_mut(production);
        Ok(())
    }
}

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
        define_production_mute(&mut grammar_table, line).map_err(|e| match e.data {
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

    Ok(grammar_table)
}

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
