use lr_core::grammar;
use proc_macro2::TokenStream;
use syn::{custom_punctuation, parse::Parse, parse_macro_input, Block, Ident};

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

struct Rule {
    lhs: Ident,
    rhs: Vec<Ident>,
    action: Block,
}

impl Rule {
    fn new(lhs: Ident, rhs: Vec<Ident>, action: Block) -> Self {
        Self { lhs, rhs, action }
    }
}

impl Parse for Rule {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        custom_punctuation!(RuleDefinitionDelimiter, ::=);

        let lhs = input.parse::<Ident>()?;
        let _ = input.parse::<RuleDefinitionDelimiter>()?;

        let mut rhs = vec![];
        loop {
            if let Ok(action) = input.parse::<Block>() {
                return Ok(Rule::new(lhs, rhs, action));
            } else {
                let next = input.parse::<Ident>()?;
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
