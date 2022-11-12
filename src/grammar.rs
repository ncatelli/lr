use std::collections::hash_map::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BuiltInTokens {
    Epsilon,
    EOF,
    EndL,
    Integer,
    Float,
    String,
    Identifier,
}

impl BuiltInTokens {
    fn as_token(&self) -> &'static str {
        match self {
            BuiltInTokens::Epsilon => "<epsilon>",
            BuiltInTokens::EOF => "<$>",
            BuiltInTokens::EndL => "<endl>",
            BuiltInTokens::Integer => "<integer>",
            BuiltInTokens::Float => "<float>",
            BuiltInTokens::String => "<string>",
            BuiltInTokens::Identifier => "<identifer>",
        }
    }
}

type ElementId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SymbolOrTokenRef {
    Symbol(ElementId),
    Token(ElementId),
}

impl std::fmt::Display for SymbolOrTokenRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymbolOrTokenRef::Symbol(id) => write!(f, "S{}", id),
            SymbolOrTokenRef::Token(id) => write!(f, "T{}", id),
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Rule {
    lhs: ElementId,
    rhs: Vec<SymbolOrTokenRef>,
}

impl Rule {
    fn new(lhs: ElementId, rhs: Vec<SymbolOrTokenRef>) -> Self {
        Self { lhs, rhs }
    }
}

impl std::fmt::Display for Rule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lhs = format!("S{}", self.lhs);
        let rhs = self
            .rhs
            .iter()
            .map(|sot| sot.to_string())
            .collect::<Vec<_>>()
            .join(" ");

        write!(f, "{} ::= {}", lhs, rhs)
    }
}

#[derive(Debug, Default)]
struct GrammarTable {
    symbols: HashMap<String, usize>,
    tokens: HashMap<String, usize>,
    rules: Vec<Rule>,
}

impl GrammarTable {
    fn new(symbols: Vec<String>, tokens: Vec<String>, rules: Vec<Rule>) -> Self {
        let symbol_iter = symbols
            .into_iter()
            .enumerate()
            .map(|(idx, symbol)| (symbol, idx + 1));
        let token_iter = tokens
            .into_iter()
            .enumerate()
            .map(|(idx, tok)| (tok, idx + 1));

        Self {
            symbols: symbol_iter.collect(),
            tokens: token_iter.collect(),
            rules,
        }
    }

    /// Adds a symbol to the table, returning its index. If the symbol already
    /// exists, the index to the previously added symbol is returned.
    fn add_symbol_mut<S: AsRef<str>>(&mut self, symbol: S) -> usize {
        let symbol = symbol.as_ref();
        let new_id = self.symbols.len() + 1;

        self.symbols.entry(symbol.to_string()).or_insert(new_id);

        // safe to unwrap due to above guarantee
        self.symbols.get(symbol).copied().unwrap()
    }

    /// Adds a token to the table, returning its index. If the token already
    /// exists, the index to the previously added token is returned.
    fn add_token_mut<S: AsRef<str>>(&mut self, token: S) -> usize {
        let token = token.as_ref();
        let new_id = self.tokens.len() + 1;

        self.tokens.entry(token.to_string()).or_insert(new_id);
        // safe to unwrap due to above guarantee
        self.tokens.get(token).copied().unwrap()
    }

    fn add_rule_mut(&mut self, rule: Rule) {
        self.rules.push(rule);
    }
}

impl std::fmt::Display for GrammarTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let header = "Grammar Table
-------------";

        let symbols = {
            let mut symbols = self.symbols.iter().collect::<Vec<_>>();
            symbols.sort_by(|(_, a), (_, b)| a.cmp(b));
            symbols
                .into_iter()
                .map(|(symbol, id)| format!("{}. '{}'\n", id, symbol))
                .collect::<String>()
        };
        let tokens = {
            let mut tokens = self.tokens.iter().collect::<Vec<_>>();
            tokens.sort_by(|(_, a), (_, b)| a.cmp(b));
            tokens
                .into_iter()
                .map(|(token, id)| format!("{}. '{}'\n", id, token))
                .collect::<String>()
        };

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

#[derive(Debug, PartialEq, Eq)]
pub enum GrammarLoadErrorKind {
    InvalidRule,
    ConflictingRule,
    Other,
}

impl std::fmt::Display for GrammarLoadErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRule => write!(f, "provided rule is invalid",),
            Self::ConflictingRule => write!(f, "provided rule conflicts with existing rule",),
            Self::Other => write!(f, "undefined load error"),
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

fn load_grammar<S: AsRef<str>>(input: S) -> Result<GrammarTable, GrammarLoadError> {
    let mut grammar_table = GrammarTable::default();

    // initial table
    let root_rule_idx = 0;
    let root_rule = Rule::new(root_rule_idx, vec![]);
    grammar_table.add_rule_mut(root_rule);

    // add default tokens
    let builtin_tokens = [
        BuiltInTokens::Epsilon,
        BuiltInTokens::EOF,
        BuiltInTokens::EndL,
        BuiltInTokens::Integer,
        BuiltInTokens::Float,
        BuiltInTokens::String,
        BuiltInTokens::Identifier,
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
        let mut rule = Rule::new(rule_id, vec![]);

        // add tokens and fill the rule.
        for elem in rhs {
            if let Some(symbol) = symbol_value_from_str(elem) {
                let symbol_id = {
                    let symbol_id = grammar_table.add_symbol_mut(symbol);
                    SymbolOrTokenRef::Symbol(symbol_id)
                };
                rule.rhs.push(symbol_id);
            // validate all token values are single length.
            } else if let Some(token) = token_value_from_str(elem) {
                let token_id = {
                    let token_id = grammar_table.add_token_mut(token);
                    SymbolOrTokenRef::Token(token_id)
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
        } else {
            grammar_table.add_rule_mut(rule)
        }
    }

    Ok(grammar_table)
}

fn symbol_value_from_str(value: &str) -> Option<&str> {
    let trimmed_value = value.trim();

    let is_wrapped = trimmed_value.starts_with('<') && trimmed_value.ends_with('>');
    let is_not_empty = trimmed_value.len() > 2;

    // guarantee that it's a symbol and that it's not just an empty symbol `<>`
    if is_wrapped && is_not_empty {
        Some(trimmed_value)
    } else {
        None
    }
}

fn token_value_from_str(value: &str) -> Option<&str> {
    let trimmed_value = value.trim();
    let value_len = trimmed_value.len();

    if value_len == 1 {
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
}
