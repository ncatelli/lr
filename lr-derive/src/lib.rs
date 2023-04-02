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
    Data, DataEnum, DeriveInput, ExprClosure, Fields, Ident, LitStr, Token,
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

impl Spanned for Production {
    fn span(&self) -> Span {
        self.0.span()
    }
}

#[derive(Debug, Clone)]
enum ReducerAction {
    Closure(ExprClosure),
    Fn(Ident),
}

enum GrammarItemAttributeKind {
    Goal,
    Production,
}

enum GrammarItemAttributeMetadata {
    Goal(GoalAttributeMetadata),
    Production(ProductionAttributeMetadata),
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

impl Spanned for GoalAttributeMetadata {
    fn span(&self) -> Span {
        let production_span = self.production.span();
        let action_span = match &self.reducer {
            ReducerAction::Closure(closure) => closure.span(),
            ReducerAction::Fn(fn_ident) => fn_ident.span(),
        };

        // Attempt to join the two spans or return the production_span if not
        // possible
        production_span.join(action_span).unwrap_or(production_span)
    }
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

impl Spanned for ProductionAttributeMetadata {
    fn span(&self) -> Span {
        let production_span = self.production.span();
        let action_span = match &self.reducer {
            ReducerAction::Closure(closure) => closure.span(),
            ReducerAction::Fn(fn_ident) => fn_ident.span(),
        };

        // Attempt to join the two spans or return the production_span if not
        // possible
        production_span.join(action_span).unwrap_or(production_span)
    }
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

            let grammar_attributes = variant.attrs.iter().filter_map(|attr| {
                if attr.path.is_ident("production") {
                    Some((GrammarItemAttributeKind::Production, attr))
                } else if attr.path.is_ident("goal") {
                    Some((GrammarItemAttributeKind::Goal, attr))
                } else {
                    None
                }
            });

            let valid_attrs_for_variant = grammar_attributes
                .map(|(kind, attr)| {
                    match kind {
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
            enum_ident: enum_variant_ident,
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

impl ReducibleGrammarTable {
    fn terminal_ident(&self) -> Option<String> {
        self.grammar_table
            .terminals()
            .filter(|t| !lr_core::grammar::BuiltinTerminals::is_builtin(t))
            .filter_map(|t| t.as_ref().split("::").next().map(|t| t.to_string()))
            .next()
    }
}

fn generate_grammer_table_from_annotated_enum(
    grammar_variants: &GrammarAnnotatedEnum,
) -> Result<ReducibleGrammarTable, String> {
    use lr_core::grammar::{
        define_production_mute, DefaultInitializedGrammarTableSansBuiltins, GrammarInitializer,
    };

    let eof_terminal = "Terminal::Eof";
    let mut grammar_table = DefaultInitializedGrammarTableSansBuiltins::initialize_table();
    let _ = grammar_table.declare_eof_terminal(eof_terminal);

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
            }
        }
    }

    if let Some(gam) = goal {
        let rhs = gam.production.value();
        let line = format!("<*> ::= {}", rhs);
        let reducer = gam.reducer.clone();

        define_production_mute(&mut grammar_table, line).map_err(|e| e.to_string())?;
        reducers.push(reducer)
    } else {
        return Err("No goal production defined".to_string());
    }

    for (non_terminal, ram) in productions {
        let rhs = ram.production.value();
        let line = format!("<{}> ::= {}", non_terminal, rhs);
        let reducer = ram.reducer.clone();

        define_production_mute(&mut grammar_table, line).map_err(|e| e.to_string())?;
        reducers.push(reducer)
    }

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
        let possible_states = (0..state_cnt).into_iter().flat_map(|state_idx| {
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

struct ActionMatcherCodeGen<'a> {
    terminal_identifier: &'a Ident,
    action_table_variants: ActionIterator<'a>,
    reducers: &'a [ReducerAction],
    rhs_lens: &'a [usize],
}

impl<'a> ActionMatcherCodeGen<'a> {
    fn new(
        terminal_identifier: &'a Ident,
        action_table_variants: ActionIterator<'a>,
        reducers: &'a [ReducerAction],
        rhs_lens: &'a [usize],
    ) -> Self {
        Self {
            terminal_identifier,
            action_table_variants,
            reducers,
            rhs_lens,
        }
    }
}

impl<'a> ToTokens for ActionMatcherCodeGen<'a> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let terminal_ident = self.terminal_identifier;
        let filtered_actions = self
            .action_table_variants
            .clone()
            .into_iter()
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
                    (#state_id, <#terminal_ident as lr_core::TerminalRepresentable>::Repr::#terminal_repr) => Ok(#action_stream),
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
            .map(|(production, (reducer, rhs_len))| (production, reducer, *rhs_len));

        let reducer_variants = reducers
            .map(|(production, reducer, rhs_len)| match reducer {
                ReducerAction::Closure(ec) => {
                    quote!(#production => (#rhs_len, (#ec)(&mut parse_ctx.element_stack)),)
                }
                ReducerAction::Fn(f) => {
                    quote!(#production => (#rhs_len, (#f)(&mut parse_ctx.element_stack)),)
                }
            })
            .collect::<TokenStream>();

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
                    input.peek().map(|term| term.to_variant_repr()).unwrap_or_else(#terminal_ident::eof),
                    current_state
                )),
                Action::Accept => {
                    let element = match parse_ctx.element_stack.len() {
                        1 => Ok(parse_ctx.element_stack.pop().unwrap()),
                        0 => Err("Reached accept state with empty stack".to_string()),
                        _ => Err(format!(
                            "Reached accept state with data on stack {:?}",
                            parse_ctx.element_stack,
                        )),
                    }?;

                    match element {
                        TerminalOrNonTerminal::Terminal(term) => {
                            return Err(format!(
                                "top of stack was a terminal at accept state: {:?}",
                                term
                            ))
                        }
                        TerminalOrNonTerminal::NonTerminal(nonterm) => return Ok(nonterm),
                    }
                }
            }?;
        );

        tokens.extend(action_matcher_stream);
        tokens.extend(action_dispatcher_stream)
    }
}

struct GotoTableLookupCodeGen<'a> {
    nonterminal_ident: Ident,
    variant_idents: Vec<Ident>,

    goto_table: &'a Vec<Vec<Goto>>,
}

impl<'a> GotoTableLookupCodeGen<'a> {
    fn new(
        nonterminal_ident: Ident,
        variant_idents: Vec<Ident>,
        goto_table: &'a Vec<Vec<Goto>>,
    ) -> Self {
        Self {
            nonterminal_ident,
            variant_idents,
            goto_table,
        }
    }
}

impl<'a> ToTokens for GotoTableLookupCodeGen<'a> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let mut variants = vec![];

        let per_non_terminal_gotos = self
            .goto_table
            .iter()
            // skip the goal non-terminal
            .skip(1)
            .enumerate();

        let nonterm_ident = &self.nonterminal_ident;

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
                    (#state, #nonterm_ident::#production_ident(_)) => Some(#goto_state),
                );
                variants.push(goto_variant_stream);
            }
        }

        // collect all match patterns into a single stream an interpolate them
        // into the lookup.
        let variants = variants.into_iter().collect::<TokenStream>();
        let lookup_goto_stream = quote!(
            fn lookup_goto(state: usize, non_term: &NonTerminal) -> Option<usize> {
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
struct ParserCtxCodeGen<'a> {
    terminal_identifier: &'a Ident,
    non_terminal_identifier: &'a Ident,
}

impl<'a> ParserCtxCodeGen<'a> {
    fn new(terminal_identifier: &'a Ident, non_terminal_identifier: &'a Ident) -> Self {
        Self {
            terminal_identifier,
            non_terminal_identifier,
        }
    }
}

impl<'a> ToTokens for ParserCtxCodeGen<'a> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let term_ident = format_ident!("{}", &self.terminal_identifier);
        let nonterm_ident = format_ident!("{}", &self.non_terminal_identifier.to_string());

        let parser_ctx_stream = quote!(
            struct ParseContext {
                state_stack: Vec<usize>,
                element_stack: Vec<lr_core::TerminalOrNonTerminal<#term_ident, #nonterm_ident>>,
            }

            impl ParseContext {
                fn push_state_mut(&mut self, state_id: usize) {
                    self.state_stack.push(state_id)
                }

                fn pop_state_mut(&mut self) -> Option<usize> {
                    self.state_stack.pop()
                }

                fn push_element_mut(&mut self, elem: TerminalOrNonTerminal<#term_ident, #nonterm_ident>) {
                    self.element_stack.push(elem)
                }

                fn pop_element_mut(
                    &mut self,
                ) -> Option<TerminalOrNonTerminal<#term_ident, #nonterm_ident>> {
                    self.element_stack.pop()
                }
            }

            impl Default for ParseContext {
                fn default() -> Self {
                    Self {
                        state_stack: vec![],
                        element_stack: vec![],
                    }
                }
            }
        );

        tokens.extend(parser_ctx_stream);
    }
}

fn codegen(
    terminal_identifier: &Ident,
    nonterminal_identifier: &Ident,
    grammar_table: &ReducibleGrammarTable,
    table: &StateTable,
) -> Result<TokenStream, String> {
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
    let action_matcher_codegen =
        ActionMatcherCodeGen::new(terminal_identifier, states_iter, reducers, &rhs_lens)
            .into_token_stream();

    let parser_ctx = ParserCtxCodeGen::new(terminal_identifier, nonterminal_identifier);

    let parser_fn_stream = quote!(
        #[allow(unused)]
        pub fn lr_parse_input<S>(input: S) -> Result<#nonterminal_identifier, String>
        where
            S: AsRef<[#terminal_identifier]>
        {
            use lr_core::lr::{Action, ProductionId, StateId};

            let mut input = input.as_ref().into_iter().copied().peekable();
            let mut parse_ctx = ParseContext::default();
            parse_ctx.push_state_mut(0);

            loop {
                let current_state = parse_ctx
                    .pop_state_mut()
                    .ok_or_else(|| "state stack is empty".to_string())?;

                let next_term_repr = input.peek().map(|term| term.to_variant_repr()).unwrap_or_else(#terminal_identifier::eof);
                #action_matcher_codegen
            }
        }
    );

    let stream = [
        parser_ctx.into_token_stream(),
        goto_table_codegen.into_token_stream(),
        parser_fn_stream.into_token_stream(),
    ]
    .into_iter()
    .collect();

    Ok(stream)
}

/// The dispatcher method for tokens annotated with the Lr1 derive.
#[proc_macro_derive(Lr1, attributes(goal, production))]
pub fn build_lr1_parser(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    use lr_core::{generate_table_from_grammar, GeneratorKind};

    let input = parse_macro_input!(input as DeriveInput);

    let annotated_enum = parse(input).unwrap();

    let reducible_grammar_table =
        generate_grammer_table_from_annotated_enum(&annotated_enum).unwrap();

    let lr_table =
        generate_table_from_grammar(GeneratorKind::Lr1, &reducible_grammar_table.grammar_table)
            .unwrap();

    let state_table = StateTable::new(&reducible_grammar_table, &lr_table);

    let term_ident = reducible_grammar_table
        .terminal_ident()
        .map(|t| format_ident!("{}", t))
        .unwrap();
    let non_terminal_ident = annotated_enum.enum_ident;

    codegen(
        &term_ident,
        &non_terminal_ident,
        &reducible_grammar_table,
        &state_table,
    )
    .unwrap()
    .into()
}
