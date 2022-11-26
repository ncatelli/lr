mod grammar;
mod lr;

pub enum GeneratorKind {
    Lr1,
}

pub fn generate_table<G: AsRef<str>>(kind: GeneratorKind, grammar: G) -> Result<(), String> {
    match kind {
        GeneratorKind::Lr1 => {
            use crate::lr::LrTableGenerator;

            crate::lr::Lr1::generate_table(grammar)
                .map(|_| ())
                .map_err(|e| e.to_string())
        }
    }
}
