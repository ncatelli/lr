# lr

A minimal LR parser generator framework designed to (optionally) work alongside [relex](https://github.com/ncatelli/relex) to build LR parsers that tightly couple with built in types, namely enums. 

## Examples
Defining grammars on an enum can be handled at compile type by specifying a set of productions, and a goal on variants of an enum.

```rust
#[derive(Debug, Lr1, PartialEq)]
pub enum NonTerminal {
    #[goal(r"<E>", reduce_e_unary_non_term)]
    #[production(r"<E> Terminal::Star <B>", |elems| { reduce_e_binary_non_term(2, elems) })]
    #[production(r"<E> Terminal::Plus <B>", |elems| { reduce_e_binary_non_term(3, elems) })]
    #[production(r"<B>", reduce_e_unary_non_term)]
    E(Box<NonTermKind>),
    #[production(r"Terminal::Int", reduce_b_non_term)]
    B(Terminal),
}
```

## Generator types
### LR(1)
Currently the generator only supports LR(1) grammars.

#### Attributes
Attributes for defining a grammar consist of both a `rule`, which contains the right-hand side of a production within the grammar and a reducer action, which can be either a closure or function following the signature of `Fn(&mut Vec<lr_core::TerminalOrNonTerminal<T, NT>) -> Result<NT, String>`. Where `T` and `NT` represent a terminal and non-terminal symbol in the grammar respectively.
##### goal
`goal` defines the final, success state of a parser. When reached the parser will return the tree as returned from this state's reducer.

##### production
`production` defines all other grammar rules. Any variant can have as many productions as are necessary. 

# Warnings

This tool was primarily built to support other projects that shared the same, no dependency goals and restrictions that I am currently working on. Use under the understanding that support for this will be best-effort.