pub use lr_derive::Lr1;
pub use relex_derive::Relex;

#[derive(Relex, Debug, PartialEq)]
pub enum Token {
    #[matches(r"-")]
    Minus,
    #[matches(r"*")]
    Star,
    #[matches(r"1")]
    One,
}

#[derive(Debug, Lr1)]
pub enum ParseTree {
    #[goal(r"<E> <$>")]
    #[rule(r"<T> Token::Minus <E>")]
    #[rule(r"<T>")]
    E,
    #[rule(r"<F> Token::Star <T>")]
    #[rule(r"<F>")]
    T,
    #[rule(r"Token::One")]
    F,
}

fn main() {}
