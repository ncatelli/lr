use lr_core::{
    grammar::GrammarTable,
    lr::{Action, Goto, LrTable},
};
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote, ToTokens};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input,
    spanned::Spanned,
    Data, DataEnum, DeriveInput, ExprClosure, Fields, Generics, Ident, LitStr, Token,
};

struct Production(LitStr);

impl Production {
    fn new(production: LitStr) -> Self {
        Self(production)
    }

    fn value(&self) -> String {
        self.0.value()
    }
}

#[derive(Debug, Clone)]
struct ParserState {
    ident: Ident,
    generics: Option<Generics>,
}

impl ParserState {
    fn new(ident: Ident, generics: Option<Generics>) -> Self {
        Self { ident, generics }
    }
}

#[derive(Debug, Clone)]
enum ReducerAction {
    Closure(ExprClosure),
    Fn(Ident),
}

enum GrammarItemAttributeKind {
    State,
    Goal,
    Production,
}

enum GrammarItemAttributeMetadata {
    State(StateAttributeMetadata),
    Goal(GoalAttributeMetadata),
    Production(ProductionAttributeMetadata),
}

impl From<StateAttributeMetadata> for GrammarItemAttributeMetadata {
    fn from(value: StateAttributeMetadata) -> Self {
        Self::State(value)
    }
}

impl From<GoalAttributeMetadata> for GrammarItemAttributeMetadata {
    fn from(value: GoalAttributeMetadata) -> Self {
        Self::Goal(value)
    }
}

impl From<ProductionAttributeMetadata> for GrammarItemAttributeMetadata {
    fn from(value: ProductionAttributeMetadata) -> Self {
        Self::Production(value)
    }
}

struct StateAttributeMetadata {
    ty: Ident,
    generics: Option<Generics>,
}

impl Parse for StateAttributeMetadata {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let lookahead = input.lookahead1();
        let ty = if lookahead.peek(Ident) {
            let ty: Ident = input.parse()?;

            ty
        } else {
            return Err(lookahead.error());
        };

        // check for type param
        let lookahead = input.lookahead1();
        // Attempt to parse out generics, otherwise default to none.
        let generics = if lookahead.peek(Token![<]) {
            let generics = input.parse::<Generics>()?;
            Some(generics)
        } else {
            None
        };

        Ok(Self { ty, generics })
    }
}

fn parse_attribute_metadata(input: ParseStream<'_>) -> syn::Result<(Production, ReducerAction)> {
    let lookahead = input.lookahead1();
    let spanned_production = if lookahead.peek(LitStr) {
        let production: LitStr = input.parse()?;

        Production::new(production)
    } else {
        return Err(lookahead.error());
    };

    let _separator: Token![,] = input.parse()?;

    // attempt to parse out a closure, and if this falls through,
    // attempt an Ident representing a function.
    let expr_closure_action = input.parse::<ExprClosure>();
    match expr_closure_action {
        Ok(action) => syn::Result::Ok((spanned_production, ReducerAction::Closure(action))),
        Err(_) => {
            let action = input.parse::<Ident>().map_err(|e| {
                let span = e.span();
                syn::Error::new(span, "expected either a closure or a function identifier")
            })?;
            syn::Result::Ok((spanned_production, ReducerAction::Fn(action)))
        }
    }
}

struct GoalAttributeMetadata {
    production: Production,
    reducer: ReducerAction,
}

impl Parse for GoalAttributeMetadata {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let (production, reducer) = parse_attribute_metadata(input)?;
        Ok(GoalAttributeMetadata {
            production,
            reducer,
        })
    }
}

struct ProductionAttributeMetadata {
    production: Production,
    reducer: ReducerAction,
}

impl Parse for ProductionAttributeMetadata {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let (production, reducer) = parse_attribute_metadata(input)?;
        Ok(ProductionAttributeMetadata {
            production,
            reducer,
        })
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
    enum_generics: Generics,
    variant_metadata: Vec<ProductionAnnotatedEnumVariant>,
}

fn parse(input: DeriveInput) -> Result<GrammarAnnotatedEnum, syn::Error> {
    let input_span = input.span();
    let enum_ident = input.ident;
    let enum_generics = input.generics;

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

            let grammar_attributes = variant.attrs.iter().filter_map(|attr| {
                if attr.path().is_ident("production") {
                    Some((GrammarItemAttributeKind::Production, attr))
                } else if attr.path().is_ident("goal") {
                    Some((GrammarItemAttributeKind::Goal, attr))
                } else if attr.path().is_ident("state") {
                    Some((GrammarItemAttributeKind::State, attr))
                } else {
                    None
                }
            });

            let valid_attrs_for_variant = grammar_attributes
                .map(|(kind, attr)| {
                    match kind {
                        GrammarItemAttributeKind::State => attr
                            .parse_args_with(StateAttributeMetadata::parse)
                            .map(GrammarItemAttributeMetadata::from),
                        GrammarItemAttributeKind::Goal => attr
                            .parse_args_with(GoalAttributeMetadata::parse)
                            .map(GrammarItemAttributeMetadata::from),
                        GrammarItemAttributeKind::Production => attr
                            .parse_args_with(ProductionAttributeMetadata::parse)
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

            // Collect all productions for a production variant.
            if !valid_attrs_for_variant.is_empty() {
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
            enum_ident,
            enum_generics,
            variant_metadata: enriched_enum_variants,
        })
}

#[derive(Debug)]
struct ReducibleGrammarTable {
    grammar_table: GrammarTable,
    reducers: Vec<ReducerAction>,
}

impl ReducibleGrammarTable {
    fn new(grammar_table: GrammarTable, reducers: Vec<ReducerAction>) -> Self {
        Self {
            grammar_table,
            reducers,
        }
    }
}

fn state_from_annotated_enum(
    annotated_enum: &GrammarAnnotatedEnum,
) -> Result<Option<ParserState>, String> {
    let mut state = None;

    for variant in &annotated_enum.variant_metadata {
        let attr_metadata = &variant.attr_metadata;
        for kind in attr_metadata {
            match kind {
                GrammarItemAttributeMetadata::State(_) if state.is_some() => {
                    return Err("multiple state items defined".to_string())
                }
                GrammarItemAttributeMetadata::State(s) => {
                    let ident = s.ty.clone();
                    let generics = s.generics.clone();

                    state = Some(ParserState::new(ident, generics))
                }
                // ignore non-metadata variants
                _ => continue,
            }
        }
    }

    Ok(state)
}

fn generate_grammar_table_from_annotated_enum(
    grammar_variants: &GrammarAnnotatedEnum,
) -> Result<ReducibleGrammarTable, String> {
    use lr_core::grammar::{
        define_production_mut, DefaultInitializedGrammarTableSansBuiltins, GrammarInitializer,
    };

    let mut grammar_table = DefaultInitializedGrammarTableSansBuiltins::initialize_table();

    let mut goal = None;
    let mut productions = vec![];
    let mut reducers = vec![];

    for production in &grammar_variants.variant_metadata {
        let attr_metadata = &production.attr_metadata;
        for production_kind in attr_metadata {
            match production_kind {
                // error if multiple goals are defined.
                GrammarItemAttributeMetadata::Goal(_) if goal.is_some() => {
                    return Err("multiple goals defined".to_string())
                }
                GrammarItemAttributeMetadata::Goal(g) => {
                    goal = Some(g);
                }
                GrammarItemAttributeMetadata::Production(ram) => {
                    let non_terminal = production.non_terminal.to_string();
                    productions.push((non_terminal, ram));
                }
                // ignore State and other metadata variants
                _ => continue,
            }
        }
    }

    if let Some(gam) = goal {
        let rhs = gam.production.value();
        let line = format!("<*> ::= {}", rhs);
        let reducer = gam.reducer.clone();

        define_production_mut(&mut grammar_table, line).map_err(|e| e.to_string())?;
        reducers.push(reducer)
    } else {
        return Err("No goal production defined".to_string());
    }

    for (non_terminal, ram) in productions {
        let rhs = ram.production.value();
        let line = format!("<{}> ::= {}", non_terminal, rhs);
        let reducer = ram.reducer.clone();

        define_production_mut(&mut grammar_table, line).map_err(|e| e.to_string())?;
        reducers.push(reducer)
    }

    let terminal_ident = grammar_table
        .terminals()
        .filter(|t| !lr_core::grammar::BuiltinTerminals::is_builtin(t))
        .filter_map(|t| t.as_ref().split("::").next().map(|t| t.to_string()))
        .next()
        .ok_or_else(|| "No terminal defined".to_string())?;

    let eof_terminal = format!("{}::Eof", terminal_ident);
    let _ = grammar_table.declare_eof_terminal(eof_terminal);
    grammar_table.finalize();

    Ok(ReducibleGrammarTable::new(grammar_table, reducers))
}

/// A wrapper type for iterating over state collection sets.
#[derive(Debug)]
struct StateTable<'a> {
    reducible_grammar_table: &'a ReducibleGrammarTable,
    state_table: &'a LrTable,
}

impl<'a> StateTable<'a> {
    fn new(grammar_table: &'a ReducibleGrammarTable, state_table: &'a LrTable) -> Self {
        Self {
            reducible_grammar_table: grammar_table,
            state_table,
        }
    }

    fn possible_states_iter(&self) -> ActionIterator<'a> {
        let grammar_table = &self.reducible_grammar_table.grammar_table;

        let state_cnt = self.state_table.states;
        let possible_states = (0..state_cnt).flat_map(|state_idx| {
            let action_table = &self.state_table.action;

            let action_columns = action_table
                .iter()
                .enumerate()
                // safe to unwrap given bounds derived from range
                .map(move |(col_idx, col)| {
                    let action = col.get(state_idx).unwrap();
                    (state_idx, col_idx, action)
                });

            action_columns
        });

        let terminals = grammar_table.terminals();
        let lookahead_variants = terminals.collect::<Vec<_>>();
        let states = possible_states
            .map(|(state_idx, lookahead_idx, action)| {
                let lookahead = lookahead_variants[lookahead_idx];

                ActionVariant::new(state_idx, lookahead, action)
            })
            .collect();

        ActionIterator::new(states)
    }
}

impl<'a> std::fmt::Display for StateTable<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let grammar_table = &self.reducible_grammar_table.grammar_table;

        write!(
            f,
            "{}\n{}",
            grammar_table,
            self.state_table.human_readable_format(grammar_table)
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct ActionVariant<'a> {
    state_id: usize,
    lookahead: lr_core::grammar::Terminal<'a>,
    action: &'a Action,
}

impl<'a> ActionVariant<'a> {
    fn new(state_id: usize, lookahead: lr_core::grammar::Terminal<'a>, action: &'a Action) -> Self {
        Self {
            state_id,
            lookahead,
            action,
        }
    }
}

/// An ordered iterator over all actions in a grammar table.
#[derive(Debug, Clone)]
struct ActionIterator<'a> {
    states: Vec<ActionVariant<'a>>,
}

impl<'a> ActionIterator<'a> {
    fn new(states: Vec<ActionVariant<'a>>) -> Self {
        Self { states }
    }
}

impl<'a> Iterator for ActionIterator<'a> {
    type Item = ActionVariant<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.states.pop()
    }
}

enum ActionMatcherState {
    Stateful,
    Stateless,
}

/// A marker trait for representing the kind of state matching code to generate.
trait ActionMatcherStateHandleable {}

struct Stateless;
impl ActionMatcherStateHandleable for Stateless {}

struct Stateful;
impl ActionMatcherStateHandleable for Stateful {}

struct ActionMatcherCodeGen<'a> {
    matcher_kind: ActionMatcherState,
    nonterminal_identifier: &'a Ident,
    action_table_variants: ActionIterator<'a>,
    reducers: &'a [ReducerAction],
    rhs_lens: &'a [usize],
}

impl<'a> ActionMatcherCodeGen<'a> {
    fn new(
        matcher_kind: ActionMatcherState,
        nonterminal_identifier: &'a Ident,
        action_table_variants: ActionIterator<'a>,
        reducers: &'a [ReducerAction],
        rhs_lens: &'a [usize],
    ) -> Self {
        Self {
            matcher_kind,
            nonterminal_identifier,
            action_table_variants,
            reducers,
            rhs_lens,
        }
    }
}

impl<'a> ToTokens for ActionMatcherCodeGen<'a> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let nonterminal_identifier = self.nonterminal_identifier;
        let filtered_actions = self
            .action_table_variants
            .clone()
            // prune out all the deadstate actions
            .filter(|variant| !matches!(variant.action, Action::DeadState));

        let variants = filtered_actions.map(
            |ActionVariant {
                 state_id,
                 lookahead,
                 action,
             }| {
                let lookahead_repr = lookahead.as_ref();

                let terminal_variant_repr = if lookahead_repr.contains("::") {
                    lookahead.as_ref().split("::").last().unwrap()
                } else {
                    lookahead_repr
                };

                // format the variant as an ident.
                let terminal_repr = format_ident!("{}", terminal_variant_repr);

                // generate the corresponding action representation for each possible state.
                let action_stream = match action {
                    Action::Accept => quote!(Action::Accept),
                    Action::Shift(state) => {
                        let state = state.as_usize();
                        quote!(Action::Shift(StateId::unchecked_new(#state)))
                    }
                    Action::Reduce(reduce_to) => {
                        let reduce_to = reduce_to.as_usize();
                        quote!(Action::Reduce(ProductionId::unchecked_new(#reduce_to)))
                    }
                    Action::DeadState => quote!(Action::DeadState),
                };

                // matches on the terminal's representation, hence the wordy
                // tuple rhs. This casts the type to it's associated
                // Repr type, which _should_ align with the shown value.
                quote!(
                    (#state_id, <<#nonterminal_identifier as lr_core::NonTerminalRepresentable>::Terminal as lr_core::TerminalRepresentable>::Repr::#terminal_repr) => Ok(#action_stream),
                )
            },
        );

        let variants = variants.collect::<TokenStream>();
        let action_matcher_stream = quote!(
            // the current state and the copyable representation of the terminal.
            let action = match (current_state, next_term_repr) {
                #variants
                _ => Err(format!(
                    "unknown parser error with: {:?}",
                    (current_state, input.peek())
                )),
            }?;
        );

        let rhs_lens = self.rhs_lens.iter();

        // generate reducer variants
        let reducers = self
            .reducers
            .iter()
            .zip(rhs_lens)
            .enumerate()
            // skip the goal production.
            .skip(1)
            .map(|(production, (reducer, rhs_len))| (production, reducer, *rhs_len));

        let reducer_variants = reducers
            .map(|(production, reducer, rhs_len)| match self.matcher_kind {
                ActionMatcherState::Stateful => match reducer {
                    ReducerAction::Closure(ec) => {
                        quote!(#production => (#rhs_len, (#ec)(&mut *parse_ctx.user_state, &mut parse_ctx.element_stack)),)
                    }
                    ReducerAction::Fn(f) => {
                        quote!(#production => (#rhs_len, (#f)(&mut *parse_ctx.user_state, &mut parse_ctx.element_stack)),)
                    }
                },
                ActionMatcherState::Stateless => match reducer {
                    ReducerAction::Closure(ec) => {
                        quote!(#production => (#rhs_len, (#ec)(&mut parse_ctx.element_stack)),)
                    }
                    ReducerAction::Fn(f) => {
                        quote!(#production => (#rhs_len, (#f)(&mut parse_ctx.element_stack)),)
                    }
                },
            })
            .collect::<TokenStream>();

        let goal_reducer = match self.matcher_kind {
            ActionMatcherState::Stateful => match &self.reducers.get(0).unwrap() {
                ReducerAction::Closure(ec) => {
                    quote!((#ec)(&mut *parse_ctx.user_state, &mut parse_ctx.element_stack))
                }
                ReducerAction::Fn(f) => {
                    quote!((#f)(&mut *parse_ctx.user_state, &mut parse_ctx.element_stack))
                }
            },
            ActionMatcherState::Stateless => match &self.reducers.get(0).unwrap() {
                ReducerAction::Closure(ec) => {
                    quote!((#ec)(&mut parse_ctx.element_stack))
                }
                ReducerAction::Fn(f) => {
                    quote!((#f)(&mut parse_ctx.element_stack))
                }
            },
        };

        let action_dispatcher_stream = quote!(
            match action {
                Action::Shift(next_state) => {
                    // a shift should never occur on an eof making this safe to unwrap.
                    let term = input.next().map(TerminalOrNonTerminal::Terminal).unwrap();
                    parse_ctx.push_element_mut(term);

                    parse_ctx.push_state_mut(current_state);
                    parse_ctx.push_state_mut(next_state.as_usize());
                    Ok(())
                }
                Action::Reduce(reduce_to) => {
                    let (rhs_len, non_term) = match reduce_to.as_usize() {
                        #reducer_variants
                        _ => (
                            0,
                            Err(format!(
                                "unable to reduce to production {}.",
                                reduce_to.as_usize()
                            )),
                        ),
                    };

                    let non_term = non_term?;

                    // peek at the last state before the nth element taken.
                    let prev_state = {
                        let mut prev_state = parse_ctx.pop_state_mut();
                        for _ in 1..rhs_len {
                            prev_state = parse_ctx.pop_state_mut();
                        }
                        let prev_state =
                            prev_state.ok_or_else(|| "state stack is empty".to_string())?;
                        parse_ctx.push_state_mut(prev_state);
                        prev_state
                    };

                    let goto_state = lookup_goto(prev_state, &non_term).ok_or_else(|| {
                        format!(
                            "no goto state for non_terminal {:?} in state {}",
                            &non_term, current_state
                        )
                    })?;

                    parse_ctx.push_state_mut(goto_state);

                    parse_ctx
                        .element_stack
                        .push(TerminalOrNonTerminal::NonTerminal(non_term));

                    Ok(())
                }
                Action::DeadState => Err(format!(
                    "unexpected input {:?} for state {}",
                    input.peek().map(|term| term.to_variant_repr()).unwrap_or_else(<<#nonterminal_identifier as lr_core::NonTerminalRepresentable>::Terminal as lr_core::TerminalRepresentable>::eof),
                    current_state
                )),
                Action::Accept => {
                    let goal = match parse_ctx.element_stack.len() {
                        1 => #goal_reducer,
                        0 => Err("Reached accept state with empty stack".to_string()),
                        _ => Err(format!(
                            "Reached accept state with data on stack {:?}",
                            parse_ctx.element_stack,
                        )),
                    }?;

                    return Ok(goal);
                }
            }?;
        );

        tokens.extend(action_matcher_stream);
        tokens.extend(action_dispatcher_stream)
    }
}

struct GotoTableLookupCodeGen<'a> {
    nonterminal_ident: Ident,
    nonterminal_generics: &'a Generics,
    variant_idents: Vec<Ident>,

    goto_table: &'a Vec<Vec<Goto>>,
}

impl<'a> GotoTableLookupCodeGen<'a> {
    fn new(
        nonterminal_ident: Ident,
        nonterminal_generics: &'a Generics,
        variant_idents: Vec<Ident>,
        goto_table: &'a Vec<Vec<Goto>>,
    ) -> Self {
        Self {
            nonterminal_ident,
            nonterminal_generics,
            variant_idents,
            goto_table,
        }
    }
}

impl<'a> ToTokens for GotoTableLookupCodeGen<'a> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let per_non_terminal_gotos = self
            .goto_table
            .iter()
            // skip the goal non-terminal
            .skip(1)
            .enumerate();

        let nonterminal_identifier = &self.nonterminal_ident;
        let nonterminal_generics = self.nonterminal_generics;
        let nonterminal_params = self.nonterminal_generics.params.clone();
        let nonterminal_signature = if !nonterminal_generics.params.is_empty() {
            quote!(#nonterminal_identifier #nonterminal_generics)
        } else {
            quote!(#nonterminal_identifier)
        };

        let mut variants = vec![];

        for (non_terminal_id, gotos) in per_non_terminal_gotos {
            let valid_variants =
                gotos
                    .iter()
                    .enumerate()
                    .filter_map(|(state_id, goto)| match goto {
                        Goto::State(goto_state) => Some((state_id, non_terminal_id, goto_state)),
                        Goto::DeadState => None,
                    });

            for (state, production_offset, goto_state) in valid_variants {
                let production_ident = &self.variant_idents[production_offset];
                let goto_variant_stream = quote!(
                    (#state, #nonterminal_identifier::#production_ident(_)) => Some(#goto_state),
                );
                variants.push(goto_variant_stream);
            }
        }

        // collect all match patterns into a single stream an interpolate them
        // into the lookup.
        let variants = variants.into_iter().collect::<TokenStream>();
        let lookup_goto_stream = quote!(
            fn lookup_goto<#nonterminal_params>(state: usize, non_term: &#nonterminal_signature) -> Option<usize> {
                match (state, non_term) {
                    #variants
                    _ => None,
                }
            }
        );

        tokens.extend(lookup_goto_stream)
    }
}

/// Generates the context for storing parse state.
struct ParserCtxCodeGen;

impl ToTokens for ParserCtxCodeGen {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let parser_ctx_stream = quote!(
            struct ParseContext<'a, STATE, T, NT> {
                /// user provide contextual state.
                user_state: &'a mut STATE,
                state_stack: Vec<usize>,
                element_stack: Vec<lr_core::TerminalOrNonTerminal<T, NT>>,
            }

            impl<'a, STATE, T, NT> ParseContext<'a, STATE, T, NT> {
                const DEFAULT_STACK_SIZE: usize = 128;

                fn new(state: &'a mut STATE) -> Self {
                    Self {
                        user_state: state,
                        state_stack: Vec::with_capacity(Self::DEFAULT_STACK_SIZE),
                        element_stack: Vec::with_capacity(Self::DEFAULT_STACK_SIZE),
                    }
                }
            }

            impl<'a, STATE, T, NT> ParseContext<'a, STATE, T, NT> {
                fn push_state_mut(&mut self, state_id: usize) {
                    self.state_stack.push(state_id)
                }

                fn pop_state_mut(&mut self) -> Option<usize> {
                    self.state_stack.pop()
                }

                fn push_element_mut(&mut self, elem: TerminalOrNonTerminal<T, NT>) {
                    self.element_stack.push(elem)
                }

                fn pop_element_mut(&mut self) -> Option<TerminalOrNonTerminal<T, NT>> {
                    self.element_stack.pop()
                }
            }
        );

        tokens.extend(parser_ctx_stream);
    }
}

fn codegen(
    nonterminal_identifier: &Ident,
    nonterminal_generics: &Generics,
    maybe_user_state: &Option<ParserState>,
    grammar_table: &ReducibleGrammarTable,
    table: &StateTable,
) -> Result<TokenStream, String> {
    let maybe_user_state_signature_and_params = maybe_user_state.as_ref().map(|state| {
        let ident = &state.ident;
        let maybe_generics = &state.generics;

        match maybe_generics {
            Some(generics) => {
                let params = generics.params.clone();
                (quote!(#ident), quote!(#params))
            }
            None => (quote!(#ident), quote!()),
        }
    });

    let nonterminal_params = nonterminal_generics.params.clone();
    let nonterminal_signature = if !nonterminal_generics.params.is_empty() {
        quote!(#nonterminal_identifier #nonterminal_generics)
    } else {
        quote!(#nonterminal_identifier)
    };

    let goto_formatted_nonterms = grammar_table
        .grammar_table
        .non_terminals()
        .skip(1)
        .map(|s| {
            s.as_ref()
                .trim_start_matches('<')
                .trim_end_matches('>')
                .to_string()
        })
        .map(|s| format_ident!("{}", s))
        .collect::<Vec<_>>();

    let goto_table_codegen = GotoTableLookupCodeGen::new(
        nonterminal_identifier.clone(),
        nonterminal_generics,
        goto_formatted_nonterms,
        &table.state_table.goto,
    );

    let states_iter = table.possible_states_iter();
    let reducers = grammar_table.reducers.as_ref();
    let rhs_lens = grammar_table
        .grammar_table
        .productions()
        .map(|production_ref| production_ref.rhs_len())
        .collect::<Vec<_>>();

    // if is stateful?
    let parseable_fn_stream = if let Some((user_state_ident, user_state_params)) =
        maybe_user_state_signature_and_params
    {
        let action_matcher_codegen = ActionMatcherCodeGen::new(
            ActionMatcherState::Stateful,
            nonterminal_identifier,
            states_iter,
            reducers,
            &rhs_lens,
        )
        .into_token_stream();

        let user_state_signature = if user_state_params.is_empty() {
            quote!(#user_state_ident)
        } else {
            quote!(#user_state_ident<#user_state_params>)
        };

        quote!(
            impl<#nonterminal_params> lr_core::LrStatefulParseable for #nonterminal_signature {
                type State = #user_state_signature;

                fn parse_input<S>(state: &mut Self::State, input: S) -> Result<Self, String>
                where
                    S: IntoIterator<Item = <Self as lr_core::NonTerminalRepresentable>::Terminal>
                {
                    use lr_core::lr::{Action, ProductionId, StateId};

                    let mut input = input.into_iter().peekable();
                    let mut parse_ctx = ParseContext::new(state);
                    parse_ctx.push_state_mut(0);

                    loop {
                        let current_state = parse_ctx
                            .pop_state_mut()
                            .ok_or_else(|| "state stack is empty".to_string())?;

                        let next_term_repr = input.peek().map(|term| term.to_variant_repr()).unwrap_or_else(|| <<Self as lr_core::NonTerminalRepresentable>::Terminal as lr_core::TerminalRepresentable>::eof());
                        #action_matcher_codegen
                    }
                }
            }
        )
    } else {
        let stateless_action_matcher_codegen = ActionMatcherCodeGen::new(
            ActionMatcherState::Stateless,
            nonterminal_identifier,
            states_iter,
            reducers,
            &rhs_lens,
        )
        .into_token_stream();

        quote!(
            impl<#nonterminal_params> lr_core::LrParseable for #nonterminal_signature {
                fn parse_input<S>(input: S) -> Result<Self, String>
                where
                    S: IntoIterator<Item = <Self as lr_core::NonTerminalRepresentable>::Terminal>
                {
                    use lr_core::lr::{Action, ProductionId, StateId};

                    let mut input = input.into_iter().peekable();
                    let mut state = ();
                    let mut parse_ctx = ParseContext::new(&mut state);
                    parse_ctx.push_state_mut(0);

                    loop {
                        let current_state = parse_ctx
                            .pop_state_mut()
                            .ok_or_else(|| "state stack is empty".to_string())?;

                        let next_term_repr = input.peek().map(|term| term.to_variant_repr()).unwrap_or_else(|| <<Self as lr_core::NonTerminalRepresentable>::Terminal as lr_core::TerminalRepresentable>::eof());
                        #stateless_action_matcher_codegen
                    }
                }
            }
        )
    };

    let stream = [
        ParserCtxCodeGen.into_token_stream(),
        goto_table_codegen.into_token_stream(),
        parseable_fn_stream.into_token_stream(),
    ]
    .into_iter()
    .collect();

    Ok(stream)
}

/// The dispatcher method for enums annotated with the Lr1 derive.
#[proc_macro_derive(Lr1, attributes(state, goal, production))]
pub fn build_lr1_parser(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    use lr_core::{generate_table_from_grammar, GeneratorKind};

    let input = parse_macro_input!(input as DeriveInput);

    let annotated_enum = parse(input).unwrap();

    let maybe_user_state: Option<ParserState> = state_from_annotated_enum(&annotated_enum).unwrap();

    let reducible_grammar_table =
        generate_grammar_table_from_annotated_enum(&annotated_enum).unwrap();

    let lr_table =
        generate_table_from_grammar(GeneratorKind::Lr1, &reducible_grammar_table.grammar_table)
            .unwrap();

    let state_table = StateTable::new(&reducible_grammar_table, &lr_table);

    let nonterminal_identifier = annotated_enum.enum_ident;
    let nonterminal_generics = annotated_enum.enum_generics;

    codegen(
        &nonterminal_identifier,
        &nonterminal_generics,
        &maybe_user_state,
        &reducible_grammar_table,
        &state_table,
    )
    .unwrap()
    .into()
}
