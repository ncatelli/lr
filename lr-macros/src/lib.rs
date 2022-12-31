use lr_core::grammar;
use proc_macro2::{Span, TokenStream};
use syn::{custom_punctuation, parse::Parse, parse_macro_input, Block, Ident, Token};

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

fn parse(input: TokenStream) -> syn::Result<Rules> {
    syn::parse2::<Rules>(input)
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
