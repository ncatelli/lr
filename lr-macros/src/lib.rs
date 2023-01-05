use std::collections::HashMap;

use lr_core::{self, grammar, lr};
use proc_macro2::TokenStream;
use syn::{parse_macro_input, spanned::Spanned, LitStr};

/// All lines signifying options are prefixed by the following delimiter.
const OPTION_PREFIX: &str = "%";

fn parse(input: TokenStream) -> syn::Result<lr_core::TableGenerationMetadata> {
    use grammar::load_grammar;
    use lr::LrTableGenerator;
    let span = input.span();

    let grammar_str = syn::parse2::<LitStr>(input)?.value();

    let (option_lines, rules) = grammar_str.lines().fold(
        (Vec::new(), Vec::new()),
        |(mut options, mut rules), line| {
            if line.trim().starts_with(OPTION_PREFIX) {
                options.push(line);
            } else {
                rules.push(line);
            }

            (options, rules)
        },
    );

    // parse out a list of options.
    let options = {
        let mut options = HashMap::new();

        for option in option_lines {
            let option = option.trim();
            let mut split_opt = option.splitn(2, ' ');

            // trims the leading `%` off a key and checks for it's existence.
            let key = split_opt
                .next()
                .map(|key| (key[1..]).trim())
                .and_then(|key| if key.is_empty() { None } else { Some(key) })
                .ok_or_else(|| format!("unknown key for option: {}", option))
                .map_err(|message| syn::Error::new(span, message))?;

            let value = split_opt
                .next()
                .map(|value| value.trim())
                .ok_or_else(|| format!("unknown value for option: {}", option))
                .map_err(|message| syn::Error::new(span, message))?;

            options.insert(key, value);
        }
        options
    };

    let grammar_table = load_grammar(&rules.join("\n"))
        .map_err(|e| lr_core::Error::new(lr_core::ErrorKind::GrammarError(e)).to_string())
        .map_err(|message| syn::Error::new(span, message))?;

    let table = lr::Lr1::generate_table(&grammar_table)
        .map_err(|e| lr_core::Error::new(lr_core::ErrorKind::TableGenerationError(e)).to_string())
        .map_err(|message| syn::Error::new(span, message))?;

    // extract options
    let ast_kind = options
        .get("type")
        .ok_or("undefined %type option")
        .map_err(|message| syn::Error::new(span, message))?;
    let token_kind = options
        .get("token")
        .ok_or("undefined %token option")
        .map_err(|message| syn::Error::new(span, message))?;

    let generation_metadata =
        lr_core::TableGenerationMetadata::new(grammar_table, table, ast_kind, token_kind);

    Ok(generation_metadata)
}

fn _codegen(metadata: &lr_core::TableGenerationMetadata) -> syn::Result<TokenStream> {
    use lr::{Action, Goto};

    let _lr_table = &metadata.lr_table;
    /*let rows = (0..lr_table.states)
    .into_iter()
    .map(|curr_state| {
        let action_row = lr_table.action.iter().map(|col| {
            col.get(curr_state)
                .map(|a| match a {
                    Action::Accept => format!("{: >10}", "accept"),
                    Action::Shift(id) => format!("{: >10}", format!("s{}", id)),
                    // rules are 1-indexed, when pretty printed
                    Action::Reduce(id) => format!("{: >10}", format!("r{}", id + 1)),
                    Action::DeadState => format!("{: >10}", DEAD_STATE_STR),
                })
                .unwrap_or_else(|| format!("{: >10}", ""))
        });
        let goto_row = self.goto.iter().skip(1).map(|col| {
            col.get(curr_state)
                .map(|g| match g {
                    Goto::State(id) => format!("{: >10}", id),
                    Goto::DeadState => format!("{: >10}", DEAD_STATE_STR),
                })
                .unwrap_or_else(|| format!("{: >10}", ""))
        });

        format!(
            "{: >6} |{}{}",
            curr_state,
            action_row.collect::<String>(),
            goto_row.collect::<String>()
        )
    })
    .collect::<Vec<_>>();*/
    todo!()
}

#[proc_macro]
pub fn parser_gen(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input);

    let generation_metadata = parse(input).unwrap();
    println!(
        "{}",
        generation_metadata
            .lr_table
            .human_readable_format(&generation_metadata.grammar_table)
    );

    proc_macro::TokenStream::new()
}
