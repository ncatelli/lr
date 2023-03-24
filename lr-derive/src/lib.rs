use proc_macro2::{Span, TokenStream};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input,
    spanned::Spanned,
    Data, DataEnum, DeriveInput, ExprClosure, Fields, Ident, LitStr, Token,
};

struct Rule(LitStr);

impl Rule {
    fn new(rule: LitStr) -> Self {
        Self(rule)
    }

    fn value(&self) -> String {
        self.0.value()
    }
}

impl Spanned for Rule {
    fn span(&self) -> Span {
        self.0.span()
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

fn parse_attribute_metadata(input: ParseStream<'_>) -> syn::Result<(Rule, ReducerAction)> {
    let lookahead = input.lookahead1();
    let spanned_rule = if lookahead.peek(LitStr) {
        let rule: LitStr = input.parse()?;

        Rule::new(rule)
    } else {
        return Err(lookahead.error());
    };

    // check whether a handler closure has been provided.
    if input.is_empty() {
        syn::Result::Ok((spanned_rule, ReducerAction::None))
    } else {
        let _separator: Token![,] = input.parse()?;

        // attempt to parse out a closure, and if this falls through,
        // attempt an Ident representing a function.
        let expr_closure_action = input.parse::<ExprClosure>();
        match expr_closure_action {
            Ok(action) => syn::Result::Ok((spanned_rule, ReducerAction::Closure(action))),
            Err(_) => {
                let action = input.parse::<Ident>().map_err(|e| {
                    let span = e.span();
                    syn::Error::new(span, "expected either a closure or a function identifier")
                })?;
                syn::Result::Ok((spanned_rule, ReducerAction::Fn(action)))
            }
        }
    }
}

struct GoalAttributeMetadata {
    rule: Rule,
    reducer: ReducerAction,
}

impl Spanned for GoalAttributeMetadata {
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

impl Parse for GoalAttributeMetadata {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let (rule, reducer) = parse_attribute_metadata(input)?;
        Ok(GoalAttributeMetadata { rule, reducer })
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

impl Parse for RuleAttributeMetadata {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let (rule, reducer) = parse_attribute_metadata(input)?;
        Ok(RuleAttributeMetadata { rule, reducer })
    }
}

/// Represents each variant in the grammar enum with it's associated productions.
struct ProductionAnnotatedEnumVariant {
    non_terminal: Ident,
    /// The productions as sourced from an attribute.
    attr_metadata: Vec<GrammarItemAttributeMetadata>,
}

/// Represents each variant of the non-terminal enum representing the grammar.
struct GrammarAnnotatedEnum {
    #[allow(unused)]
    span: Span,
    /// Represents the Identifier for the Token enum.
    #[allow(unused)]
    enum_ident: Ident,
    variant_metadata: Vec<ProductionAnnotatedEnumVariant>,
}

fn parse(input: DeriveInput) -> Result<GrammarAnnotatedEnum, syn::Error> {
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
                    // check that either field has exactly zero or one
                    //attribute
                    .and_then(|giak| {
                        match variant_fields.clone() {
                            // an unamed struct with one field
                            Fields::Unnamed(f) if f.unnamed.len() == 1 => Ok(giak),
                            // an empty filed
                            Fields::Unit => Ok(giak),
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
                Ok(ProductionAnnotatedEnumVariant {
                    non_terminal: variant_ident,
                    attr_metadata: valid_attrs_for_variant,
                })
            } else {
                Err(syn::Error::new(
                    variant_span,
                    "expect atleast one attribute is specified",
                ))
            }
        })
        .collect::<Result<_, _>>()
        .map(|enriched_enum_variants| GrammarAnnotatedEnum {
            span: input_span,
            enum_ident: enum_variant_ident,
            variant_metadata: enriched_enum_variants,
        })
}

fn generate_grammer_table_from_annotated_enum(
    grammar_variants: &GrammarAnnotatedEnum,
) -> Result<lr_core::grammar::GrammarTable, String> {
    use lr_core::grammar::{
        define_rule_mut, DefaultInitializedGrammarTableBuilder, GrammarInitializer,
    };

    let mut grammar_table = DefaultInitializedGrammarTableBuilder::initialize_table();
    let mut goal = None;
    let mut rules = vec![];

    for production in &grammar_variants.variant_metadata {
        let attr_metadata = &production.attr_metadata;
        for rule in attr_metadata {
            match rule {
                // error if multiple goals are defined.
                GrammarItemAttributeMetadata::Goal(_) if goal.is_some() => {
                    return Err("multiple goals defined".to_string())
                }
                GrammarItemAttributeMetadata::Goal(g) => {
                    goal = Some(g);
                }
                GrammarItemAttributeMetadata::Rule(ram) => {
                    let non_terminal = production.non_terminal.to_string();
                    rules.push((non_terminal, ram));
                }
            }
        }
    }

    if let Some(gam) = goal {
        let rhs = gam.rule.value();
        let line = format!("<*> ::= {}", rhs);
        define_rule_mut(&mut grammar_table, line).map_err(|e| e.to_string())?;
    } else {
        return Err("No goal production defined".to_string());
    }

    for (non_terminal, ram) in rules {
        let rhs = ram.rule.value();
        let line = format!("<{}> ::= {}", non_terminal, rhs);
        define_rule_mut(&mut grammar_table, line).map_err(|e| e.to_string())?;
    }

    Ok(grammar_table)
}

/// The dispatcher method for tokens annotated with the Lr1 derive.
#[proc_macro_derive(Lr1, attributes(goal, rule))]
pub fn relex(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let annotated_enum = parse(input).unwrap();
    let grammar_table = generate_grammer_table_from_annotated_enum(&annotated_enum).unwrap();

    println!("{}", &grammar_table);

    Ok(TokenStream::new())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
