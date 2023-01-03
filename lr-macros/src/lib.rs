use proc_macro2::{Span, TokenStream};
use syn::{custom_punctuation, parse::Parse, parse_macro_input, Block, Ident, LitStr, Token};

struct Rules(Vec<Rule>);

impl Rules {
    fn new(rules: Vec<Rule>) -> Self {
        Self(rules)
    }
}

impl Parse for Rules {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut rules = vec![];
        loop {
            if input.is_empty() {
                return Ok(Rules::new(rules));
            } else {
                let rule = input.parse::<Rule>()?;
                rules.push(rule);
            }
        }
    }
}

impl AsRef<[Rule]> for Rules {
    fn as_ref(&self) -> &[Rule] {
        &self.0
    }
}

struct SymbolIdent {
    span: Span,
    ident: Ident,
}

impl Parse for SymbolIdent {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let span = input.span();

        let _ = input.parse::<Token![<]>()?;
        let sym = input.parse::<Ident>()?;
        let _ = input.parse::<Token![>]>()?;

        Ok(SymbolIdent { span, ident: sym })
    }
}

struct TokenIdent {
    span: Span,
    ident: Ident,
}

impl Parse for TokenIdent {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let span = input.span();

        let tok = input.parse::<Ident>()?;

        Ok(TokenIdent { span, ident: tok })
    }
}

enum SymbolOrTokenIdent {
    Symbol(SymbolIdent),
    Token(TokenIdent),
}

impl Parse for SymbolOrTokenIdent {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let span = input.span();
        let lookahead = input.lookahead1();

        if lookahead.peek(Token![<]) {
            SymbolIdent::parse(input).map(SymbolOrTokenIdent::Symbol)
        } else {
            TokenIdent::parse(input).map(SymbolOrTokenIdent::Token)
        }
    }
}

struct Rule {
    lhs: SymbolIdent,
    rhs: Vec<SymbolOrTokenIdent>,
    action: Block,
}

impl Rule {
    fn new(lhs: SymbolIdent, rhs: Vec<SymbolOrTokenIdent>, action: Block) -> Self {
        Self { lhs, rhs, action }
    }
}

impl Parse for Rule {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        custom_punctuation!(RuleDefinitionDelimiter, ::=);

        let lhs = input.parse::<SymbolIdent>()?;
        let _ = input.parse::<RuleDefinitionDelimiter>()?;

        let mut rhs = vec![];
        loop {
            if let Ok(action) = input.parse::<Block>() {
                return Ok(Rule::new(lhs, rhs, action));
            } else {
                let next = input.parse::<SymbolOrTokenIdent>()?;
                rhs.push(next);
            }
        }
    }
}

use grammar::GrammarLoadError;

mod grammar;
mod lr;

/// Represents the kind of table that can be generated
enum GeneratorKind {
    /// LR(1) Grammar
    Lr1,
}

#[derive(Debug, PartialEq, Eq)]
enum ErrorKind {
    GrammarError(GrammarLoadError),
    TableGenerationError(lr::TableGenError),
}

impl std::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GrammarError(err) => write!(f, "grammar error: {}", err),
            Self::TableGenerationError(err) => write!(f, "table generation error: {}", err),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Error {
    kind: ErrorKind,
    data: Option<String>,
}

impl Error {
    pub(crate) fn new(kind: ErrorKind) -> Self {
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

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.data {
            Some(ctx) => write!(f, "{}: {}", &self.kind, ctx),
            None => write!(f, "{}", &self.kind),
        }
    }
}

#[allow(unused)]
fn generate_table<G: AsRef<str>>(kind: GeneratorKind, grammar: G) -> Result<lr::LrTable, Error> {
    use grammar::load_grammar;

    let grammar = grammar.as_ref();
    let grammar_table =
        load_grammar(grammar).map_err(|e| Error::new(ErrorKind::GrammarError(e)))?;

    match kind {
        GeneratorKind::Lr1 => {
            use crate::lr::LrTableGenerator;

            crate::lr::Lr1::generate_table(&grammar_table)
                .map_err(|e| Error::new(ErrorKind::TableGenerationError(e)))
        }
    }
}

fn parse(input: TokenStream) -> syn::Result<Rules> {
    let grammar = syn::parse2::<LitStr>(input);

    todo!()
}

fn codegen<R: AsRef<[Rule]>>(rules: R) -> syn::Result<TokenStream> {
    let _rules = rules.as_ref();

    todo!()
}

#[proc_macro]
pub fn parser_gen(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input);

    let _rules = parse(input).unwrap();

    //println!("{:#?}", rules);
    proc_macro::TokenStream::new()
}
