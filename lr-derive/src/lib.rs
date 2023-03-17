use proc_macro2::{Span, TokenStream};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input,
    spanned::Spanned,
    Data, DataEnum, DeriveInput, ExprClosure, Fields, Ident, LitStr, Token,
};

struct Rule {
    rule: LitStr,
}

impl Rule {
    fn new(rule: LitStr) -> Self {
        Self { rule }
    }
}

impl Spanned for Rule {
    fn span(&self) -> Span {
        self.rule.span()
    }
}

enum ReducerAction {
    Closure(ExprClosure),
    Fn(Ident),
    None,
}

enum GrammarItemAttributeKind {
    Goal,
    Rule,
}

enum GrammarItemAttributeMetadata {
    Goal(GoalAttributeMetadata),
    Rule(RuleAttributeMetadata),
}

impl From<GoalAttributeMetadata> for GrammarItemAttributeMetadata {
    fn from(value: GoalAttributeMetadata) -> Self {
        Self::Goal(value)
    }
}

impl From<RuleAttributeMetadata> for GrammarItemAttributeMetadata {
    fn from(value: RuleAttributeMetadata) -> Self {
        Self::Rule(value)
    }
}

struct GoalAttributeMetadata;

impl Parse for GoalAttributeMetadata {
    fn parse(_input: ParseStream) -> syn::Result<Self> {
        Ok(GoalAttributeMetadata)
    }
}

struct RuleAttributeMetadata {
    rule: Rule,
    reducer: ReducerAction,
}

impl Spanned for RuleAttributeMetadata {
    fn span(&self) -> Span {
        let rule_span = self.rule.span();
        let action_span = match &self.reducer {
            ReducerAction::Closure(closure) => Some(closure.span()),
            ReducerAction::Fn(fn_ident) => Some(fn_ident.span()),
            ReducerAction::None => None,
        };

        // Attempt to join the two spans or return the rule_span if not
        // possible
        action_span
            .and_then(|action_span| rule_span.join(action_span))
            .unwrap_or_else(|| rule_span)
    }
}

impl RuleAttributeMetadata {
    fn new(rule: Rule, reducer: ReducerAction) -> Self {
        Self { rule, reducer }
    }
}

impl Parse for RuleAttributeMetadata {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let lookahead = input.lookahead1();
        let spanned_rule = if lookahead.peek(LitStr) {
            let rule: LitStr = input.parse()?;

            Rule::new(rule)
        } else {
            return Err(lookahead.error());
        };

        // check whether a handler closure has been provided.
        if input.is_empty() {
            syn::Result::Ok(RuleAttributeMetadata {
                rule: spanned_rule,
                reducer: ReducerAction::None,
            })
        } else {
            let _separator: Token![,] = input.parse()?;

            // attempt to parse out a closure, and if this falls through,
            // attempt an Ident representing a function.
            let expr_closure_action = input.parse::<ExprClosure>();
            match expr_closure_action {
                Ok(action) => syn::Result::Ok(RuleAttributeMetadata {
                    rule: spanned_rule,
                    reducer: ReducerAction::Closure(action),
                }),
                Err(_) => {
                    let action = input.parse::<Ident>().map_err(|e| {
                        let span = e.span();
                        syn::Error::new(span, "expected either a closure or a function identifier")
                    })?;
                    syn::Result::Ok(RuleAttributeMetadata {
                        rule: spanned_rule,
                        reducer: ReducerAction::Fn(action),
                    })
                }
            }
        }
    }
}

/// Stores all supported lexer attributes for a given Token variant.
struct ProductionMetadata {
    non_terminal: Ident,
    attr_metadata: Vec<GrammarItemAttributeMetadata>,
}

impl ProductionMetadata {
    fn new(ident: Ident, attr_metadata: Vec<GrammarItemAttributeMetadata>) -> Self {
        Self {
            non_terminal: ident,
            attr_metadata,
        }
    }
}

/// Represents each variant of the non-terminal enum representing the grammar.
struct GrammarVariants {
    span: Span,
    /// Represents the Identifier for the Token enum.
    enum_ident: Ident,
    variant_metadata: Vec<ProductionMetadata>,
}

impl GrammarVariants {
    fn new(span: Span, enum_ident: Ident, variant_metadata: Vec<ProductionMetadata>) -> Self {
        Self {
            span,
            enum_ident,
            variant_metadata,
        }
    }
}

fn parse(input: DeriveInput) -> Result<GrammarVariants, syn::Error> {
    let input_span = input.span();
    let enum_variant_ident = input.ident;
    let enum_variants = match input.data {
        Data::Enum(DataEnum { variants, .. }) => variants,
        _ => {
            return Err(syn::Error::new(
                input_span,
                "derive macro only works on enums",
            ))
        }
    };

    // token enum variants with their tokenizer metadata parsed.
    enum_variants
        .into_iter()
        .map(|variant| {
            let variant_span = variant.span();
            let variant_ident = variant.ident;
            let variant_fields = variant.fields;

            let grammar_rule_attributes = variant.attrs.iter().filter_map(|attr| {
                if attr.path.is_ident("rule") {
                    Some((GrammarItemAttributeKind::Rule, attr))
                } else if attr.path.is_ident("goal") {
                    Some((GrammarItemAttributeKind::Goal, attr))
                } else {
                    None
                }
            });

            let valid_attrs_for_variant = grammar_rule_attributes
                .map(|(kind, attr)| {
                    match kind {
                        GrammarItemAttributeKind::Goal => attr
                            .parse_args_with(GoalAttributeMetadata::parse)
                            .map(GrammarItemAttributeMetadata::from),
                        GrammarItemAttributeKind::Rule => attr
                            .parse_args_with(RuleAttributeMetadata::parse)
                            .map(GrammarItemAttributeMetadata::from),
                    }
                    .and_then(|ram| {
                        match variant_fields.clone() {
                            // an unamed struct with one field
                            Fields::Unnamed(f) if f.unnamed.len() == 1 => Ok(ram),
                            // an empty filed
                            Fields::Unit => Ok(ram),
                            l => Err(syn::Error::new(
                                l.span(),
                                format!(
                                    "variant({}) expects exactly 1 unnamed field, got {}",
                                    &variant_ident,
                                    l.len()
                                ),
                            )),
                        }
                    })
                })
                // terminate if any attributes are invalid
                .collect::<Result<Vec<_>, _>>()?;

            // Collect all rules for a production variant.
            if valid_attrs_for_variant.len() >= 1 {
                Ok(ProductionMetadata::new(
                    variant_ident,
                    valid_attrs_for_variant,
                ))
            } else {
                Err(syn::Error::new(
                    variant_span,
                    "expect atleast one attribute is specified",
                ))
            }
        })
        .collect::<Result<_, _>>()
        .map(|enriched_enum_variants| {
            GrammarVariants::new(input_span, enum_variant_ident, enriched_enum_variants)
        })
}

/// The dispatcher method for tokens annotated with the Lr1 derive.
#[proc_macro_derive(Lr1, attributes(goal, rule))]
pub fn relex(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    parse(input)
        .map(|_| TokenStream::new())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
