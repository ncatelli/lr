use proc_macro2::{Span, TokenStream};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input,
    spanned::Spanned,
    Data, DataEnum, DeriveInput, ExprClosure, Fields, Ident, LitStr, Token,
};

struct Rule(String);

impl Rule {
    fn new(pattern: String) -> Self {
        Self(pattern)
    }
}

enum ReducerAction {
    Closure(ExprClosure),
    Fn(Ident),
    None,
}

enum AttributeKind {
    Goal,
    Rule,
}

struct RuleAttributeMetadata {
    rule: Rule,
    reducer: ReducerAction,
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
            let pat: LitStr = input.parse()?;
            let span = pat.span();

            /*
            regex_compiler::parser::parse(format!("({})", &pat.value()))
                .map(|regex| SpannedRegex::new(span, regex))
                .map_err(|e| syn::Error::new_spanned(pat, e))?
                */
            todo!()
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
struct NonTerminalVariantMetadata {
    ident: Ident,
    attr_metadata: RuleAttributeMetadata,
}

impl NonTerminalVariantMetadata {
    fn new(ident: Ident, attr_metadata: RuleAttributeMetadata) -> Self {
        Self {
            ident,
            attr_metadata,
        }
    }
}

struct GrammarVariants {
    span: Span,
    /// Represents the Identifier for the Token enum.
    enum_ident: Ident,
    variant_metadata: Vec<NonTerminalVariantMetadata>,
}

impl GrammarVariants {
    fn new(
        span: Span,
        enum_ident: Ident,
        variant_metadata: Vec<NonTerminalVariantMetadata>,
    ) -> Self {
        Self {
            span,
            enum_ident,
            variant_metadata,
        }
    }
}

fn parse(input: DeriveInput) -> Result<GrammarVariants, syn::Error> {
    let input_span = input.span();
    let tok_enum_name = input.ident;
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

            let attr_kind = variant
                .attrs
                .iter()
                .filter_map(|attr| {
                    if attr.path.is_ident("rule") {
                        Some((AttributeKind::Rule, attr))
                    } else if attr.path.is_ident("goal") {
                        Some((AttributeKind::Goal, attr))
                    } else {
                        None
                    }
                })
                .map(|(kind, attr)| match kind {
                    AttributeKind::Goal => todo!(),
                    AttributeKind::Rule => {
                        attr.parse_args_with(RuleAttributeMetadata::parse)
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
                    }
                });

            let mut variant_match_attr = attr_kind.collect::<Result<Vec<_>, _>>()?;
            if variant_match_attr.len() == 1 {
                let attr = variant_match_attr.pop().unwrap();
                Ok(NonTerminalVariantMetadata::new(variant_ident, attr))
            } else {
                Err(syn::Error::new(
                    variant_span,
                    "expect exactly one match attribute specified",
                ))
            }
        })
        .collect::<Result<_, _>>()
        .map(|enriched_token_variants| {
            GrammarVariants::new(input_span, tok_enum_name, enriched_token_variants)
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
