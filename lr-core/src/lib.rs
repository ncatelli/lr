use core::hash::Hash;

pub trait TerminalVariant<'a>: Copy + Eq + Hash + Ord + TryFrom<&'a str> {
    fn from_str_repr(src: &'a str) -> Option<Self> {
        Self::try_from(src).ok()
    }
}

/// A trait signifying that a type can be represented as a Terminal within the
/// grammar.
pub trait TerminalRepresentable<'a>
where
    Self: Sized,
    Self::VariantRepr: TerminalVariant<'a>,
{
    type VariantRepr;

    const EPSILON_VARIANT: Self::VariantRepr;
    const EOF_VARIANT: Self::VariantRepr;

    fn variant(&self) -> Self::VariantRepr;
}
