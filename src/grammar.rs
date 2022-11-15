use std::collections::hash_map::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinTokens {
    Epsilon,
    Eof,
    EndL,
    Integer,
    Float,
    String,
    Identifier,
}

impl BuiltinTokens {
    pub(crate) fn is_builtin<S: AsRef<str>>(token_str: S) -> bool {
        let val = token_str.as_ref();
        [
            Self::Epsilon,
            Self::Eof,
            Self::EndL,
            Self::Integer,
            Self::Float,
            Self::String,
            Self::Identifier,
        ]
        .iter()
        .map(|builtin| builtin.as_token())
        .any(|builtin| builtin == val)
    }

    pub(crate) fn as_token(&self) -> &'static str {
        match self {
            BuiltinTokens::Epsilon => "<epsilon>",
            BuiltinTokens::Eof => "<$>",
            BuiltinTokens::EndL => "<endl>",
            BuiltinTokens::Integer => "<integer>",
            BuiltinTokens::Float => "<float>",
            BuiltinTokens::String => "<string>",
            BuiltinTokens::Identifier => "<identifer>",
        }
    }
}

type ElementId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SymbolOrToken<'a> {
    Symbol(Symbol<'a>),
    Token(Token<'a>),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RuleRef {
    pub lhs: SymbolRef,
    pub rhs: Vec<SymbolOrTokenRef>,
}

impl RuleRef {
    fn new(lhs: SymbolRef, rhs: Vec<SymbolOrTokenRef>) -> Option<Self> {
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
pub(crate) struct Symbol<'a>(&'a str);

impl<'a> Symbol<'a> {
    pub(crate) fn new(symbol: &'a str) -> Self {
        Self(symbol)
    }
}
impl<'a> AsRef<str> for Symbol<'a> {
    fn as_ref(&self) -> &str {
        self.0
    }
}

impl<'a> std::fmt::Display for Symbol<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.0)
    }
}

/// A wrapper type for tokens borrowed from the grammar table.
#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Token<'a>(&'a str);

impl<'a> Token<'a> {
    pub(crate) fn new(token: &'a str) -> Self {
        Self(token)
    }
}

impl<'a> AsRef<str> for Token<'a> {
    fn as_ref(&self) -> &str {
        self.0
    }
}

impl<'a> From<&'a str> for Token<'a> {
    fn from(val: &'a str) -> Self {
        Token::new(val)
    }
}

impl<'a> From<BuiltinTokens> for Token<'a> {
    fn from(val: BuiltinTokens) -> Self {
        Token::new(val.as_token())
    }
}

impl<'a> std::fmt::Display for Token<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.0)
    }
}

#[derive(Debug, Default, PartialEq)]
pub(crate) struct GrammarTable {
    symbols: HashMap<String, usize>,
    tokens: HashMap<String, usize>,
    rules: Vec<RuleRef>,
}

impl GrammarTable {
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

    pub(crate) fn symbols(&self) -> SymbolIterator {
        SymbolIterator::new(self)
    }

    pub(crate) fn tokens(&self) -> TokenIterator {
        TokenIterator::new(self)
    }

    pub(crate) fn rules(&self) -> impl Iterator<Item = &RuleRef> {
        self.rules.iter()
    }
}

impl std::fmt::Display for GrammarTable {
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
pub(crate) struct SymbolIterator<'a> {
    symbols: Vec<&'a str>,
}

impl<'a> SymbolIterator<'a> {
    fn new(grammar_table: &'a GrammarTable) -> Self {
        let mut values = grammar_table
            .symbols
            .iter()
            .map(|(key, value)| (key.as_str(), value))
            .collect::<Vec<_>>();
        // reverse the order so first rule pops off the back first.
        values.sort_by(|(_, a), (_, b)| b.cmp(a));

        Self {
            symbols: values.into_iter().map(|(key, _)| key).collect(),
        }
    }
}

impl<'a> Iterator for SymbolIterator<'a> {
    type Item = Symbol<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.symbols.pop().map(Symbol)
    }
}

/// An ordered iterator over all tokens in a grammar table.
pub(crate) struct TokenIterator<'a> {
    tokens: Vec<&'a str>,
}

impl<'a> TokenIterator<'a> {
    fn new(grammar_table: &'a GrammarTable) -> Self {
        let mut values = grammar_table
            .tokens
            .iter()
            .map(|(key, value)| (key.as_str(), value))
            .collect::<Vec<_>>();
        // reverse the order so first rule pops off the back first.
        values.sort_by(|(_, a), (_, b)| b.cmp(a));

        Self {
            tokens: values.into_iter().map(|(key, _)| key).collect(),
        }
    }
}

impl<'a> Iterator for TokenIterator<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.tokens.pop().map(Token)
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

pub(crate) fn load_grammar<S: AsRef<str>>(input: S) -> Result<GrammarTable, GrammarLoadError> {
    let mut grammar_table = GrammarTable::default();

    // initial table
    let root_rule_idx = SymbolRef::new(GrammarTable::ROOT_RULE_IDX);
    let root_rule = RuleRef::new_unchecked(root_rule_idx, vec![]);
    grammar_table.add_rule_mut(root_rule);

    // add default tokens
    let builtin_tokens = [
        BuiltinTokens::Epsilon,
        BuiltinTokens::Eof,
        BuiltinTokens::EndL,
        BuiltinTokens::Integer,
        BuiltinTokens::Float,
        BuiltinTokens::String,
        BuiltinTokens::Identifier,
    ];

    for builtin_tokens in builtin_tokens {
        let symbol_string_repr = builtin_tokens.as_token().to_string();
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
    let is_builtin = BuiltinTokens::is_builtin(trimmed_value);

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
    let is_builtin = BuiltinTokens::is_builtin(trimmed_value);

    if value_len == 1 || is_builtin {
        Some(trimmed_value)
    } else {
        None
    }
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
        let grammar_table = load_grammar(TEST_GRAMMAR);

        assert!(grammar_table.is_ok());

        // safe to unwrap with assertion.
        let grammar_table = grammar_table.unwrap();

        assert_eq!(1, grammar_table.symbols.len());
        assert_eq!(9, grammar_table.tokens.len());
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

        let mut symbol_iter = grammar_table.symbols();

        assert_eq!(Some(Symbol("<expr>")), symbol_iter.next());
        assert_eq!(Some(Symbol("<addition>")), symbol_iter.next());
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
            .filter(|token| !token.0.starts_with('<'));

        assert_eq!(Some(Token("(")), token_iter.next());
        assert_eq!(Some(Token(")")), token_iter.next());
        assert_eq!(Some(Token("+")), token_iter.next());
        assert_eq!(None, token_iter.next());
    }
}
