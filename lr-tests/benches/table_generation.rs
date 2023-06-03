use criterion::{criterion_group, criterion_main, Criterion};

fn lr1_table_generation_from_known_grammar(c: &mut Criterion) {
    let grammar = "
                <Expression> ::= <Assignment>
                <Assignment> ::= <Conditional>
                <Conditional> ::= <LogicalOr>
                <LogicalOr> ::= <LogicalAnd>
                <LogicalAnd> ::= <InclusiveOr>
                <InclusiveOr> ::= <ExclusiveOr>
                <ExclusiveOr> ::= <And>
                <And> ::= <Equality>
                <Equality> ::= <Relational>
                <Relational> ::= <Shift>
                <Shift> ::= <Additive>
                <Additive> ::= <Multiplicative>
                <Multiplicative> ::= <Cast>
                <Cast> ::= <Unary>
                <Unary> ::= <Postfix>
                <Unary> ::= Token::PlusPlus <Unary>
                <Unary> ::= Token::MinusMinus <Unary>
                <Unary> ::= <UnaryOperator> <Cast>
                <UnaryOperator> ::= Token::Ampersand
                <UnaryOperator> ::= Token::Star
                <UnaryOperator> ::= Token::Plus
                <UnaryOperator> ::= Token::Minus
                <UnaryOperator> ::= Token::Tilde
                <UnaryOperator> ::= Token::Bang
                <Postfix> ::= <Primary>
                <Postfix> ::= <Postfix> Token::LeftBracket <Expression> Token::RightBracket
                <Postfix> ::= <Postfix> Token::LeftParen Token::RightParen
                <Postfix> ::= <Postfix> Token::LeftParen <ArgumentExpressionList> Token::RightParen
                <Postfix> ::= <Postfix> Token::Dot Token::Identifier
                <Postfix> ::= <Postfix> Token::Arrow Token::Identifier
                <Postfix> ::= <Postfix> Token::PlusPlus
                <Postfix> ::= <Postfix> Token::MinusMinus
                <ArgumentExpressionList> ::= <Assignment>
                <Primary> ::= Token::Identifier
                <Primary> ::= <Constant>
                <Primary> ::= Token::StringLiteral
                <Primary> ::= Token::LeftParen <Expression> Token::RightParen
                <Constant> ::= Token::IntegerConstant
                <Constant> ::= Token::CharacterConstant
                <Constant> ::= Token::FloatingConstant
                ";

    c.bench_function("lr1 table generation", |b| {
        b.iter(|| {
            let table =
                lr_core::generate_table_from_production_set(lr_core::GeneratorKind::Lr1, grammar)
                    .unwrap();

            assert_eq!(table.states, 141)
        });
    });
}

criterion_group!(benches, lr1_table_generation_from_known_grammar);
criterion_main!(benches);
