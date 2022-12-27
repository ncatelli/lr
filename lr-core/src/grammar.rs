use std::collections::hash_map::HashMap;
use std::hash::Hash;

pub const STRING_GOAL_REPR: &str = "<*>";
pub const STRING_EOF_REPR: &str = "<$>";
pub const STRING_EPSILON_REPR: &str = "<epsilon>";

/// A trait signifying that a type can be represented as a Terminal within the
/// grammar.
pub trait TerminalRepresentable
where
    Self: Sized,
    Self::VariantRepr: Copy + Eq + Hash + Ord,
{
    type VariantRepr;

    fn variant(&self) -> Self::VariantRepr;
    fn epsilon_variant() -> Self::VariantRepr;
    fn eof_variant() -> Self::VariantRepr;
}

/// A trait signifying that a type can be represented as a NonTerminal within
/// the grammar.
pub trait NonTerminalRepresentable
where
    Self: Sized,
    Self::VariantRepr: Copy + Eq + Hash + Ord,
{
    type VariantRepr;

    fn variant(&self) -> Self::VariantRepr;
    /// Attempts to convert a string representation to a corresponding nonterminal kind.
    fn goal_variant() -> Self::VariantRepr;
}

pub trait GrammarElementHumanReadableRepresentable: std::fmt::Display {
    fn human_readable_repr(&self) -> String {
        self.to_string()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SymbolOrToken<NT, T> {
    Symbol(NT),
    Token(T),
}

impl<NT, T> std::fmt::Display for SymbolOrToken<NT, T>
where
    NT: std::fmt::Display,
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymbolOrToken::Symbol(s) => s.fmt(f),
            SymbolOrToken::Token(t) => t.fmt(f),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReferenceSymbolOrToken<'a, S, T> {
    Symbol(ReferenceSymbol<'a, S>),
    Token(ReferenceToken<'a, T>),
}

impl<'a, S, T> std::fmt::Display for ReferenceSymbolOrToken<'a, S, T>
where
    S: std::fmt::Display,
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReferenceSymbolOrToken::Symbol(s) => s.fmt(f),
            ReferenceSymbolOrToken::Token(t) => t.fmt(f),
        }
    }
}

/// A wrapper type for symbols that reference the grammar table.
#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SymbolRef(usize);

impl SymbolRef {
    pub(crate) fn new(symbol: usize) -> Self {
        Self(symbol)
    }

    pub(crate) fn as_usize(&self) -> usize {
        self.0
    }
}

impl From<usize> for SymbolRef {
    fn from(val: usize) -> Self {
        Self::new(val)
    }
}

impl std::fmt::Display for SymbolRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Adds 1 as production is 1 indexed in human-readable format but 0
        // indexed internally. This is probably going to haunt me at some
        // point.
        write!(f, "S{}", &self.0 + 1)
    }
}

/// A wrapper type for tokens that reference the grammar table.
#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TokenRef(usize);

impl TokenRef {
    pub(crate) fn new(token: usize) -> Self {
        Self(token)
    }

    pub(crate) fn as_usize(&self) -> usize {
        self.0
    }
}

impl From<usize> for TokenRef {
    fn from(val: usize) -> Self {
        Self::new(val)
    }
}

impl std::fmt::Display for TokenRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Adds 1 as production is 1 indexed in human-readable format but 0
        // indexed internally. This is probably going to haunt me at some
        // point.
        write!(f, "T{}", &self.0 + 1)
    }
}

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SymbolOrTokenRef {
    Symbol(SymbolRef),
    Token(TokenRef),
}

impl std::fmt::Display for SymbolOrTokenRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymbolOrTokenRef::Symbol(id) => write!(f, "{}", id),
            SymbolOrTokenRef::Token(id) => write!(f, "{}", id),
        }
    }
}

#[derive(Debug, Hash, Clone, PartialEq, Eq)]
pub(crate) struct RuleRef {
    pub lhs: SymbolRef,
    pub rhs: Vec<SymbolOrTokenRef>,
}

impl RuleRef {
    pub(crate) fn new(lhs: SymbolRef, rhs: Vec<SymbolOrTokenRef>) -> Option<Self> {
        let rule = Self::new_unchecked(lhs, rhs);

        if rule.is_valid() {
            Some(rule)
        } else {
            None
        }
    }

    fn new_unchecked(lhs: SymbolRef, rhs: Vec<SymbolOrTokenRef>) -> Self {
        Self { lhs, rhs }
    }

    fn is_valid(&self) -> bool {
        let is_non_terminal_rule = self
            .rhs
            .iter()
            .all(|items| items == &SymbolOrTokenRef::Symbol(self.lhs));

        !(self.rhs.is_empty() || is_non_terminal_rule)
    }
}

impl std::fmt::Display for RuleRef {
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

/// A wrapper type for symbols borrowed from the grammar table.
#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReferenceSymbol<'a, S>
where
    S: 'a,
{
    _lifetime: std::marker::PhantomData<&'a ()>,
    sym: S,
}

impl<'a, S> ReferenceSymbol<'a, S> {
    pub(crate) fn new(symbol: S) -> Self {
        Self {
            _lifetime: std::marker::PhantomData,
            sym: symbol,
        }
    }
}

impl<'a> AsRef<str> for ReferenceSymbol<'a, &'a str> {
    fn as_ref(&self) -> &str {
        self.sym
    }
}

impl<'a, S: std::fmt::Display> std::fmt::Display for ReferenceSymbol<'a, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.sym)
    }
}

/// A wrapper type for tokens borrowed from the grammar table.
#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ReferenceToken<'a, T>
where
    T: 'a,
{
    _lifetime: std::marker::PhantomData<&'a ()>,
    tok: T,
}

impl<'a, T> ReferenceToken<'a, T> {
    pub(crate) fn new(token: T) -> Self {
        Self {
            _lifetime: std::marker::PhantomData,
            tok: token,
        }
    }
}

impl<'a> AsRef<str> for ReferenceToken<'a, &'a str> {
    fn as_ref(&self) -> &str {
        self.tok
    }
}

impl<'a, T: std::fmt::Display> std::fmt::Display for ReferenceToken<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.tok)
    }
}

#[derive(Debug, PartialEq)]
pub(crate) struct GrammarTable<NT, T>
where
    NT: NonTerminalRepresentable,
    T: TerminalRepresentable,
{
    _nonterm_ty: std::marker::PhantomData<NT>,
    _term_ty: std::marker::PhantomData<T>,
    nonterminals: HashMap<NT::VariantRepr, usize>,
    terminals: HashMap<T::VariantRepr, usize>,
    rules: Vec<RuleRef>,
}

impl<S: NonTerminalRepresentable, T: TerminalRepresentable> GrammarTable<S, T> {
    pub(crate) const ROOT_RULE_IDX: usize = 0;

    /// Instantiates a new instance of a grammar table with a defined eof and
    /// epsilon terminal and goal non-terminal.
    pub(crate) fn new() -> Self {
        let grammar_table = Self::default();
        grammar_table
            .add_nonterminal(S::goal_variant())
            .add_terminal(T::epsilon_variant())
            .add_terminal(T::eof_variant())
    }

    /// Adds a symbol to the table, returning its index. If the symbol already
    /// exists, the index to the previously added symbol is returned.
    pub(crate) fn add_nonterminal_mut(&mut self, symbol: S::VariantRepr) -> usize {
        let new_id = self.nonterminals.len();
        self.nonterminals.entry(symbol).or_insert(new_id);

        // safe to unwrap due to above guarantee
        self.nonterminals.get(&symbol).copied().unwrap()
    }

    /// Idempotently adds a nonterminal to a grammar table, returning the
    /// grammar table.
    pub(crate) fn add_nonterminal(mut self, symbol: S::VariantRepr) -> Self {
        self.add_nonterminal_mut(symbol);
        self
    }

    /// Adds a token to the table, returning its index. If the token already
    /// exists, the index to the previously added token is returned.
    pub(crate) fn add_terminal_mut(&mut self, terminal: T::VariantRepr) -> usize {
        let new_id = self.terminals.len();

        self.terminals.entry(terminal).or_insert(new_id);
        // safe to unwrap due to above guarantee
        self.terminals.get(&terminal).copied().unwrap()
    }

    /// Idempotently adds a terminal to a grammar table, returning the grammar
    /// table.
    pub(crate) fn add_terminal(mut self, terminal: T::VariantRepr) -> Self {
        self.add_terminal_mut(terminal);
        self
    }

    pub(crate) fn add_rule_mut(&mut self, rule: RuleRef) {
        self.rules.push(rule);
    }

    pub(crate) fn add_rule(mut self, rule: RuleRef) -> Self {
        self.add_rule_mut(rule);
        self
    }

    pub(crate) fn nonterminals(&self) -> NonTerminalIterator<S> {
        NonTerminalIterator::new(self)
    }

    pub(crate) fn terminals(&self) -> TerminalIterator<T> {
        TerminalIterator::new(self)
    }

    pub(crate) fn nonterminal_mapping(&self, symbol: &S::VariantRepr) -> Option<SymbolRef> {
        self.nonterminals.get(symbol).map(|id| SymbolRef(*id))
    }

    pub(crate) fn terminal_mapping(&self, token: &T::VariantRepr) -> Option<TokenRef> {
        self.terminals.get(token).map(|id| TokenRef(*id))
    }

    pub(crate) fn rules(&self) -> impl Iterator<Item = &RuleRef> {
        self.rules.iter()
    }
}

impl<N: NonTerminalRepresentable, T: TerminalRepresentable> Default for GrammarTable<N, T> {
    fn default() -> Self {
        Self {
            _nonterm_ty: Default::default(),
            _term_ty: Default::default(),
            nonterminals: Default::default(),
            terminals: Default::default(),
            rules: Default::default(),
        }
    }
}

impl<S, T> std::fmt::Display for GrammarTable<S, T>
where
    S: NonTerminalRepresentable,
    T: TerminalRepresentable,
    S::VariantRepr: GrammarElementHumanReadableRepresentable,
    T::VariantRepr: GrammarElementHumanReadableRepresentable,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let header = "Grammar Table
-------------";

        let symbols = self
            .nonterminals()
            .enumerate()
            .map(|(id, symbol)| format!("{}. '{}'\n", id + 1, symbol.human_readable_repr()))
            .collect::<String>();
        let tokens = self
            .terminals()
            .enumerate()
            .map(|(id, token)| format!("{}. '{}'\n", id + 1, token.human_readable_repr()))
            .collect::<String>();

        let rules = self
            .rules
            .iter()
            .enumerate()
            // 1-indexed
            .map(|(idx, rule)| (idx + 1, rule))
            .map(|(idx, rule)| format!("{}. {}\n", idx, rule))
            .collect::<String>();

        write!(
            f,
            "{}\nSYMBOLS\n{}\nTOKENS\n{}\nRULES\n{}",
            header, symbols, tokens, rules
        )
    }
}

#[derive(Debug, Default, PartialEq)]
pub(crate) struct StringGrammarTable<S, T>
where
    S: Hash + Eq,
    T: Hash + Eq,
{
    symbols: HashMap<S, usize>,
    tokens: HashMap<T, usize>,
    rules: Vec<RuleRef>,
}

impl StringGrammarTable<String, String> {
    pub(crate) const ROOT_RULE_IDX: usize = 0;

    /// Adds a symbol to the table, returning its index. If the symbol already
    /// exists, the index to the previously added symbol is returned.
    fn add_symbol_mut<S: AsRef<str>>(&mut self, symbol: S) -> usize {
        let symbol = symbol.as_ref();
        let new_id = self.symbols.len();

        self.symbols.entry(symbol.to_string()).or_insert(new_id);

        // safe to unwrap due to above guarantee
        self.symbols.get(symbol).copied().unwrap()
    }

    /// Adds a token to the table, returning its index. If the token already
    /// exists, the index to the previously added token is returned.
    fn add_token_mut<S: AsRef<str>>(&mut self, token: S) -> usize {
        let token = token.as_ref();
        let new_id = self.tokens.len();

        self.tokens.entry(token.to_string()).or_insert(new_id);
        // safe to unwrap due to above guarantee
        self.tokens.get(token).copied().unwrap()
    }

    fn add_rule_mut(&mut self, rule: RuleRef) {
        self.rules.push(rule);
    }

    pub(crate) fn symbols(&self) -> StringSymbolIterator<String> {
        StringSymbolIterator::new(self)
    }

    pub(crate) fn tokens(&self) -> StringTokenIterator<String> {
        StringTokenIterator::new(self)
    }

    pub(crate) fn symbol_mapping<'a>(
        &self,
        symbol: &ReferenceSymbol<&'a str>,
    ) -> Option<SymbolRef> {
        self.symbols.get(symbol.sym).map(|id| SymbolRef(*id))
    }

    pub(crate) fn token_mapping<'a>(&self, token: &ReferenceToken<&'a str>) -> Option<TokenRef> {
        self.tokens.get(token.tok).map(|id| TokenRef(*id))
    }

    pub(crate) fn rules(&self) -> impl Iterator<Item = &RuleRef> {
        self.rules.iter()
    }
}

impl std::fmt::Display for StringGrammarTable<String, String> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let header = "Grammar Table
-------------";

        let symbols = self
            .symbols()
            .enumerate()
            .map(|(id, symbol)| format!("{}. '{}'\n", id + 1, symbol))
            .collect::<String>();
        let tokens = self
            .tokens()
            .enumerate()
            .map(|(id, token)| format!("{}. '{}'\n", id + 1, token))
            .collect::<String>();

        let rules = self
            .rules
            .iter()
            .enumerate()
            // 1-indexed
            .map(|(idx, rule)| (idx + 1, rule))
            .map(|(idx, rule)| format!("{}. {}\n", idx, rule))
            .collect::<String>();

        write!(
            f,
            "{}\nSYMBOLS\n{}\nTOKENS\n{}\nRULES\n{}",
            header, symbols, tokens, rules
        )
    }
}

/// An ordered iterator over all symbols in a grammar table.
pub(crate) struct StringSymbolIterator<'a, S> {
    symbols: Vec<&'a S>,
}

impl<'a> StringSymbolIterator<'a, String> {
    fn new<T: Hash + Eq>(grammar_table: &'a StringGrammarTable<String, T>) -> Self {
        let mut values = grammar_table.symbols.iter().collect::<Vec<_>>();
        // reverse the order so first rule pops off the back first.
        values.sort_by(|(_, a), (_, b)| b.cmp(a));

        Self {
            symbols: values.into_iter().map(|(key, _)| key).collect(),
        }
    }
}

impl<'a> Iterator for StringSymbolIterator<'a, String> {
    type Item = ReferenceSymbol<'a, &'a str>;

    fn next(&mut self) -> Option<Self::Item> {
        self.symbols
            .pop()
            .map(|s| s.as_str())
            .map(ReferenceSymbol::new)
    }
}

/// An ordered iterator over all symbols in a grammar table.
pub(crate) struct NonTerminalIterator<NT: NonTerminalRepresentable> {
    symbols: Vec<NT::VariantRepr>,
}

impl<'a, NT: NonTerminalRepresentable> NonTerminalIterator<NT> {
    fn new<T: TerminalRepresentable>(grammar_table: &'a GrammarTable<NT, T>) -> Self
    where
        Vec<NT::VariantRepr>: std::iter::FromIterator<NT::VariantRepr>,
    {
        let mut values = grammar_table
            .nonterminals
            .iter()
            .map(|(&lhs, &rhs)| (lhs, rhs))
            .collect::<Vec<_>>();
        // reverse the order so first rule pops off the back first.
        values.sort_by(|(_, a), (_, b)| b.cmp(a));

        Self {
            symbols: values.into_iter().map(|(key, _)| key).collect(),
        }
    }
}

impl<NT: NonTerminalRepresentable> Iterator for NonTerminalIterator<NT> {
    type Item = NT::VariantRepr;

    fn next(&mut self) -> Option<Self::Item> {
        self.symbols.pop()
    }
}

/// An ordered iterator over all tokens in a grammar table.
pub(crate) struct StringTokenIterator<'a, T> {
    tokens: Vec<&'a T>,
}

impl<'a> StringTokenIterator<'a, String> {
    fn new<S: Hash + Eq>(grammar_table: &'a StringGrammarTable<S, String>) -> Self {
        let mut values = grammar_table.tokens.iter().collect::<Vec<_>>();
        // reverse the order so first rule pops off the back first.
        values.sort_by(|(_, a), (_, b)| b.cmp(a));

        Self {
            tokens: values.into_iter().map(|(key, _)| key).collect(),
        }
    }
}

impl<'a> Iterator for StringTokenIterator<'a, String> {
    type Item = ReferenceToken<'a, &'a str>;

    fn next(&mut self) -> Option<Self::Item> {
        self.tokens
            .pop()
            .map(|s| s.as_str())
            .map(ReferenceToken::new)
    }
}

/// An ordered iterator over all tokens in a grammar table.
pub(crate) struct TerminalIterator<T: TerminalRepresentable> {
    tokens: Vec<T::VariantRepr>,
}

impl<'a, T: TerminalRepresentable> TerminalIterator<T> {
    fn new<NT: NonTerminalRepresentable>(grammar_table: &'a GrammarTable<NT, T>) -> Self
    where
        Vec<T::VariantRepr>: std::iter::FromIterator<T::VariantRepr>,
    {
        let mut values = grammar_table
            .terminals
            .iter()
            .map(|(&lhs, &rhs)| (lhs, rhs))
            .collect::<Vec<_>>();
        // reverse the order so first rule pops off the back first.
        values.sort_by(|(_, a), (_, b)| b.cmp(a));

        Self {
            tokens: values.into_iter().map(|(key, _)| key).collect(),
        }
    }
}

impl<T> Iterator for TerminalIterator<T>
where
    T: TerminalRepresentable,
{
    type Item = T::VariantRepr;

    fn next(&mut self) -> Option<Self::Item> {
        self.tokens.pop()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum GrammarLoadErrorKind {
    NoTerminalProduction,
    InvalidRule,
    ConflictingRule,
}

impl std::fmt::Display for GrammarLoadErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoTerminalProduction => {
                write!(f, "grammar does not include a terminal production")
            }
            Self::InvalidRule => write!(f, "provided rule is invalid",),
            Self::ConflictingRule => write!(f, "provided rule conflicts with existing rule",),
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

pub(crate) fn load_grammar<S: AsRef<str>>(
    input: S,
) -> Result<StringGrammarTable<String, String>, GrammarLoadError> {
    let mut grammar_table = StringGrammarTable::default();

    // initial table
    let root_rule_idx = SymbolRef::new(StringGrammarTable::ROOT_RULE_IDX);
    let root_rule = RuleRef::new_unchecked(root_rule_idx, vec![]);
    grammar_table.add_rule_mut(root_rule);
    grammar_table.add_symbol_mut(STRING_GOAL_REPR);

    // add default tokens
    let builtin_tokens = [STRING_EPSILON_REPR, STRING_EOF_REPR];

    for builtin_token in builtin_tokens {
        let symbol_string_repr = builtin_token;
        grammar_table.add_token_mut(symbol_string_repr);
    }

    // breakup input into enumerated lines.
    let lines = input
        .as_ref()
        .lines()
        .enumerate()
        .map(|(lineno, line)| (lineno + 1, line));

    let lines_containing_rules = lines
        // ignore commented lines.
        .filter(|(_, line)| !line.starts_with(';'))
        // ignore empty lines.
        .filter(|(_, line)| !line.chars().all(|c| c.is_whitespace()));

    for (lineno, line) in lines_containing_rules {
        let trimmed_line = line.trim();

        // validate the start of the line is a symbol
        if !trimmed_line.starts_with('<') {
            return Err(GrammarLoadError::new(GrammarLoadErrorKind::InvalidRule)
                .with_data(format!("lineno {}: doesn't start with symbol", lineno)));
        }

        // split the line at the assignment delimiter
        let mut split_line = trimmed_line.split("::=").collect::<Vec<_>>();
        if split_line.len() != 2 {
            return Err(
                GrammarLoadError::new(GrammarLoadErrorKind::InvalidRule).with_data(format!(
                    "lineno {}: does not contain right-hand side",
                    lineno
                )),
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

        // retrieve the LHS symbol.
        let lhs_symbol = symbol_value_from_str(lhs).ok_or_else(|| {
            GrammarLoadError::new(GrammarLoadErrorKind::InvalidRule)
                .with_data(format!("lineno {}: doesn't start with symbol", lineno))
        })?;

        let rule_id = grammar_table.add_symbol_mut(lhs_symbol);
        let rule_ref = SymbolRef::from(rule_id);
        let mut rule = RuleRef::new_unchecked(rule_ref, vec![]);

        // add tokens and fill the rule.
        for elem in rhs {
            if let Some(symbol) = symbol_value_from_str(elem) {
                let symbol_id = {
                    let symbol_id = grammar_table.add_symbol_mut(symbol);
                    let symbol_ref = SymbolRef::new(symbol_id);

                    SymbolOrTokenRef::Symbol(symbol_ref)
                };
                rule.rhs.push(symbol_id);
            // validate all token values are single length.
            } else if let Some(token) = token_value_from_str(elem) {
                let token_id = {
                    let token_id = grammar_table.add_token_mut(token);
                    let token_ref = TokenRef::from(token_id);

                    SymbolOrTokenRef::Token(token_ref)
                };
                rule.rhs.push(token_id);
            } else {
                return Err(GrammarLoadError::new(GrammarLoadErrorKind::InvalidRule)
                    .with_data(format!("lineno {}: invalid rhs value ({})", lineno, elem)));
            }
        }

        if grammar_table.rules.contains(&rule) {
            return Err(GrammarLoadError::new(GrammarLoadErrorKind::ConflictingRule)
                .with_data(format!("lineno {}: {} ", lineno, &rule)));
        } else if !rule.is_valid() {
            return Err(GrammarLoadError::new(GrammarLoadErrorKind::InvalidRule)
                .with_data(format!("lineno {}: {} ", lineno, &rule)));
        } else {
            grammar_table.add_rule_mut(rule)
        }
    }

    // add the first production to the goal.
    let root_production = grammar_table
        .rules()
        .next()
        .map(|rule_ref| rule_ref.lhs)
        .ok_or_else(|| GrammarLoadError::new(GrammarLoadErrorKind::NoTerminalProduction))?;
    let first_non_root_production = grammar_table
        .rules()
        .nth(1)
        .map(|rule_ref| rule_ref.lhs)
        .map(SymbolOrTokenRef::Symbol)
        .ok_or_else(|| GrammarLoadError::new(GrammarLoadErrorKind::NoTerminalProduction))?;

    grammar_table.rules[root_production.as_usize()]
        .rhs
        .push(first_non_root_production);

    Ok(grammar_table)
}

fn symbol_value_from_str(value: &str) -> Option<&str> {
    let trimmed_value = value.trim();

    let is_wrapped = trimmed_value.starts_with('<') && trimmed_value.ends_with('>');
    let is_not_empty = trimmed_value.len() > 2;
    let is_builtin = trimmed_value == STRING_EOF_REPR || trimmed_value == STRING_EPSILON_REPR;

    // guarantee that it's a symbol and that it's not just an empty symbol `<>`
    if is_wrapped && is_not_empty && !is_builtin {
        Some(trimmed_value)
    } else {
        None
    }
}

fn token_value_from_str(value: &str) -> Option<&str> {
    let trimmed_value = value.trim();
    let value_len = trimmed_value.len();

    let is_builtin = trimmed_value == STRING_EOF_REPR || trimmed_value == STRING_EPSILON_REPR;

    if value_len == 1 || is_builtin {
        Some(trimmed_value)
    } else {
        None
    }
}

pub(crate) fn rule_ref_from_parts<NT, T, RHS>(
    grammar_table: &GrammarTable<NT, T>,
    lhs: NT::VariantRepr,
    rhs: RHS,
) -> Result<RuleRef, GrammarLoadError>
where
    NT: NonTerminalRepresentable,
    T: TerminalRepresentable,
    NT::VariantRepr: std::fmt::Debug,
    T::VariantRepr: std::fmt::Debug,
    RHS: AsRef<[SymbolOrToken<NT::VariantRepr, T::VariantRepr>]>,
{
    let lhs_ref = grammar_table.nonterminal_mapping(&lhs).ok_or_else(|| {
        GrammarLoadError::new(GrammarLoadErrorKind::InvalidRule)
            .with_data(format!("invalid lhs nonterminal: {:?}", &lhs))
    })?;
    let rhs_refs = rhs
        .as_ref()
        .iter()
        .copied()
        .map(|sot| {
            match sot {
                SymbolOrToken::Symbol(nonterminal) => grammar_table
                    .nonterminal_mapping(&nonterminal)
                    .map(SymbolOrTokenRef::Symbol),
                SymbolOrToken::Token(terminal) => grammar_table
                    .terminal_mapping(&terminal)
                    .map(SymbolOrTokenRef::Token),
            }
            .ok_or_else(|| {
                GrammarLoadError::new(GrammarLoadErrorKind::InvalidRule)
                    .with_data(format!("invalid grammar variant: {:?}", &sot))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    RuleRef::new(lhs_ref, rhs_refs)
        .ok_or_else(|| GrammarLoadError::new(GrammarLoadErrorKind::InvalidRule))
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! nonterm_variant {
        ($variant:expr) => {
            SymbolOrToken::Symbol($variant)
        };
    }

    macro_rules! term_variant {
        ($variant:expr) => {
            SymbolOrToken::Token($variant)
        };
    }

    const TEST_GRAMMAR: &str = "
; a comment
<parens> ::= ( <parens> )
<parens> ::= ( )
<parens> ::= <parens> ( )
<parens> ::= <parens> ( <parens> )
<parens> ::= ( ) <parens>
<parens> ::= ( <parens> ) <parens>
";

    use enum_variant_kind_derive::EnumVariantKind;

    #[derive(EnumVariantKind, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum ParensGrammarToken {
        Epsilon,
        Eof,
        LeftParen,
        RightParen,
    }

    impl TerminalRepresentable for ParensGrammarToken {
        type VariantRepr = ParensGrammarTokenKind;

        fn eof_variant() -> Self::VariantRepr {
            Self::VariantRepr::Eof
        }

        fn epsilon_variant() -> Self::VariantRepr {
            Self::VariantRepr::Epsilon
        }

        fn variant(&self) -> Self::VariantRepr {
            match self {
                ParensGrammarToken::Epsilon => Self::VariantRepr::Epsilon,
                ParensGrammarToken::Eof => Self::VariantRepr::Eof,
                ParensGrammarToken::LeftParen => Self::VariantRepr::LeftParen,
                ParensGrammarToken::RightParen => Self::VariantRepr::RightParen,
            }
        }
    }

    #[derive(EnumVariantKind, Debug)]
    enum ParensGrammarSymbol {
        Goal,
        Parens,
    }

    impl NonTerminalRepresentable for ParensGrammarSymbol {
        type VariantRepr = ParensGrammarSymbolKind;

        fn goal_variant() -> Self::VariantRepr {
            Self::VariantRepr::Goal
        }

        fn variant(&self) -> Self::VariantRepr {
            match self {
                ParensGrammarSymbol::Goal => Self::VariantRepr::Goal,
                ParensGrammarSymbol::Parens => Self::VariantRepr::Parens,
            }
        }
    }

    #[test]
    fn should_parse_table_with_valid_test_grammar() {
        let grammar_table = {
            let mut grammar_table = GrammarTable::<ParensGrammarSymbol, ParensGrammarToken>::new()
                .add_nonterminal(ParensGrammarSymbolKind::Parens)
                .add_terminal(ParensGrammarTokenKind::LeftParen)
                .add_terminal(ParensGrammarTokenKind::RightParen);

            let rules = [
                (
                    ParensGrammarSymbol::goal_variant(),
                    vec![
                        nonterm_variant!(ParensGrammarSymbolKind::Parens),
                        term_variant!(ParensGrammarToken::eof_variant()),
                    ],
                ),
                (
                    ParensGrammarSymbolKind::Parens,
                    vec![
                        nonterm_variant!(ParensGrammarSymbolKind::Parens),
                        term_variant!(ParensGrammarTokenKind::LeftParen),
                        nonterm_variant!(ParensGrammarSymbolKind::Parens),
                        term_variant!(ParensGrammarTokenKind::RightParen),
                    ],
                ),
                (
                    ParensGrammarSymbolKind::Parens,
                    vec![
                        nonterm_variant!(ParensGrammarSymbolKind::Parens),
                        term_variant!(ParensGrammarTokenKind::LeftParen),
                        term_variant!(ParensGrammarTokenKind::RightParen),
                    ],
                ),
                (
                    ParensGrammarSymbolKind::Parens,
                    vec![
                        nonterm_variant!(ParensGrammarSymbolKind::Parens),
                        nonterm_variant!(ParensGrammarSymbolKind::Parens),
                        term_variant!(ParensGrammarTokenKind::LeftParen),
                        term_variant!(ParensGrammarTokenKind::RightParen),
                    ],
                ),
                (
                    ParensGrammarSymbolKind::Parens,
                    vec![
                        nonterm_variant!(ParensGrammarSymbolKind::Parens),
                        nonterm_variant!(ParensGrammarSymbolKind::Parens),
                        term_variant!(ParensGrammarTokenKind::LeftParen),
                        nonterm_variant!(ParensGrammarSymbolKind::Parens),
                        term_variant!(ParensGrammarTokenKind::RightParen),
                    ],
                ),
                (
                    ParensGrammarSymbolKind::Parens,
                    vec![
                        nonterm_variant!(ParensGrammarSymbolKind::Parens),
                        term_variant!(ParensGrammarTokenKind::LeftParen),
                        term_variant!(ParensGrammarTokenKind::RightParen),
                        nonterm_variant!(ParensGrammarSymbolKind::Parens),
                    ],
                ),
                (
                    ParensGrammarSymbolKind::Parens,
                    vec![
                        term_variant!(ParensGrammarTokenKind::LeftParen),
                        nonterm_variant!(ParensGrammarSymbolKind::Parens),
                        term_variant!(ParensGrammarTokenKind::RightParen),
                        nonterm_variant!(ParensGrammarSymbolKind::Parens),
                    ],
                ),
            ];

            let rule_refs = rules
                .into_iter()
                .map(|(lhs, rhs)| rule_ref_from_parts(&grammar_table, lhs, &rhs))
                .collect::<Result<Vec<_>, _>>()
                .unwrap();

            for rule_ref in rule_refs {
                grammar_table.add_rule_mut(rule_ref);
            }

            grammar_table
        };

        assert_eq!(2, grammar_table.nonterminals.len());
        // 2 builtins plus `(` and `)`
        assert_eq!(4, grammar_table.terminals.len());
        assert_eq!(7, grammar_table.rules.len());
    }

    #[test]
    fn should_parse_table_with_valid_string_test_grammar() {
        let grammar_table = load_grammar(TEST_GRAMMAR);

        assert!(grammar_table.is_ok());

        // safe to unwrap with assertion.
        let grammar_table = grammar_table.unwrap();

        assert_eq!(2, grammar_table.symbols.len());
        // 2 builtins plus `(` and `)`
        assert_eq!(4, grammar_table.tokens.len());
        assert_eq!(7, grammar_table.rules.len());
    }

    #[test]
    fn should_error_on_invalid_rule() {
        let res = load_grammar(
            "
<expr> ::= ( <expr> )
<expr> ::=  abcd
        ",
        );

        assert_eq!(
            Err(GrammarLoadErrorKind::InvalidRule),
            res.map_err(|e| e.kind)
        );
    }

    #[test]
    fn should_error_on_conflicting_rule() {
        let res = load_grammar(
            "
<expr> ::= ( <expr> )
<expr> ::= ( )
<expr> ::= ( ) 
        ",
        );

        assert_eq!(
            Err(GrammarLoadErrorKind::ConflictingRule),
            res.map_err(|e| e.kind)
        );
    }

    #[test]
    fn should_iterate_symbols_in_order() {
        let res = load_grammar(
            "
<expr> ::= ( <expr> )
<expr> ::= <addition>
<addition> ::= <expr> + <expr>  
        ",
        );

        assert!(res.is_ok());
        let grammar_table = res.unwrap();
        let goal_symbol_repr = STRING_GOAL_REPR;

        let mut symbol_iter = grammar_table.symbols();

        assert_eq!(
            Some(ReferenceSymbol::new(goal_symbol_repr)),
            symbol_iter.next()
        );
        assert_eq!(Some(ReferenceSymbol::new("<expr>")), symbol_iter.next());
        assert_eq!(Some(ReferenceSymbol::new("<addition>")), symbol_iter.next());
        assert_eq!(None, symbol_iter.next());
    }

    #[test]
    fn should_iterate_tokens_in_order() {
        let res = load_grammar(
            "
<expr> ::= ( <expr> )
<expr> ::= <addition>
<addition> ::= <expr> + <expr>  
        ",
        );

        assert!(res.is_ok());
        let grammar_table = res.unwrap();

        let mut token_iter = grammar_table
            .tokens()
            // strip out the builtins for the sake of testing.
            .filter(|token| !token.tok.starts_with('<'));

        assert_eq!(Some(ReferenceToken::new("(")), token_iter.next());
        assert_eq!(Some(ReferenceToken::new(")")), token_iter.next());
        assert_eq!(Some(ReferenceToken::new("+")), token_iter.next());
        assert_eq!(None, token_iter.next());
    }
}
