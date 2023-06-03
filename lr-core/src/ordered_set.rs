use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{BuildHasher, Hash, Hasher};

/// A hasher that takes and returns 8 bytes as a u64.
///
/// # Caller Asserts
///
/// This returns the first 8 bytes from the array. Largely expecting the value
/// to be a pre-generated hash.
struct NoOpU64Hasher(u64);

impl Hasher for NoOpU64Hasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        const MAX_BYTES: usize = (u64::BITS / 8) as usize;

        let len = MAX_BYTES.min(bytes.len());

        let mut buf = [0u8; MAX_BYTES];
        buf[..len].copy_from_slice(&bytes[..len]);

        self.0 = u64::from_ne_bytes(buf);
    }
}

#[derive(Debug, Clone, Copy)]
struct NoOpHashBuilder;

impl BuildHasher for NoOpHashBuilder {
    type Hasher = NoOpU64Hasher;

    fn build_hasher(&self) -> Self::Hasher {
        NoOpU64Hasher(0)
    }
}

#[derive(Debug, Clone)]
pub struct OrderedSet<T: Hash> {
    elem_idx: HashMap<u64, usize, NoOpHashBuilder>,
    elems: Vec<T>,
}

impl<T: Hash> OrderedSet<T> {
    pub fn new() -> Self {
        let elem_idx = HashMap::with_hasher(NoOpHashBuilder);

        Self {
            elem_idx,
            elems: Default::default(),
        }
    }

    #[allow(unused)]
    pub fn with_capacity(capacity: usize) -> Self {
        let elem_idx = HashMap::with_capacity_and_hasher(capacity, NoOpHashBuilder);

        Self {
            elem_idx,
            elems: Vec::with_capacity(capacity),
        }
    }

    /// Returns a boolean signifying if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.elems.is_empty()
    }

    /// Returns the number of elements in the set.
    pub fn len(&self) -> usize {
        self.elems.len()
    }

    /// Insert an element into a set, returning `true` if the item was not
    /// previously a member of the set.
    pub fn insert(&mut self, elem: T) -> bool {
        // this causes double hashing but it's fine for the sake of storage.
        let mut hasher = DefaultHasher::default();
        elem.hash(&mut hasher);
        let elem_hash = hasher.finish();

        let slot = self.elems.len();

        if let std::collections::hash_map::Entry::Vacant(e) = self.elem_idx.entry(elem_hash) {
            e.insert(slot);
            self.elems.push(elem);

            true
        } else {
            false
        }
    }

    /// Clears all elements from the ordered set.
    #[allow(unused)]
    pub fn clear(&mut self) {
        self.elem_idx.clear();
        self.elems.clear();
    }

    /// Returns the position of a given element is a member the set, otherwise
    /// `None` is returned.
    pub fn position(&self, elem: &T) -> Option<usize> {
        let mut hasher = DefaultHasher::default();
        elem.hash(&mut hasher);
        let elem_hash = hasher.finish();

        self.elem_idx.get(&elem_hash).copied()
    }

    /// Returns a boolean signifying if a given element is a member of the set.
    #[allow(unused)]
    pub fn contains(&self, elem: &T) -> bool {
        self.position(elem).is_some()
    }

    /// Returns the [OrderedSet<T>] as a [Vec<T>] preserving the order.
    pub fn into_vec(self) -> Vec<T> {
        self.elems
    }

    /// Generates an immutable iterator over the ordered values of the set.
    pub fn iter(&self) -> std::slice::Iter<T> {
        self.elems.iter()
    }
}

impl<T: Hash> AsRef<[T]> for OrderedSet<T> {
    fn as_ref(&self) -> &[T] {
        &self.elems
    }
}

impl<T: Hash> AsMut<[T]> for OrderedSet<T> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.elems
    }
}

impl<T: Hash + PartialEq> PartialEq for OrderedSet<T> {
    fn eq(&self, other: &Self) -> bool {
        self.elems == other.elems
    }
}

impl<T: Hash + Eq> Eq for OrderedSet<T> {}

impl<T: Hash> IntoIterator for OrderedSet<T> {
    type Item = T;

    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.elems.into_iter()
    }
}

impl<T: Hash> FromIterator<T> for OrderedSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut set = OrderedSet::default();

        for elem in iter {
            set.insert(elem);
        }

        set
    }
}

impl<T: Hash> Hash for OrderedSet<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.elems.hash(state);
    }
}

impl<T: Hash> From<OrderedSet<T>> for Vec<T> {
    fn from(value: OrderedSet<T>) -> Self {
        value.into_vec()
    }
}

impl<T: Hash> Default for OrderedSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn should_generate_identity_for_all_inputs_to_no_op_hasher() {
        for i in 0..256 {
            let mut no_op_hasher = NoOpHashBuilder.build_hasher();
            let mut default_hasher = DefaultHasher::default();

            i.hash(&mut default_hasher);
            let expected_hash = default_hasher.finish();

            expected_hash.hash(&mut no_op_hasher);

            assert_eq!(no_op_hasher.finish(), expected_hash);
        }
    }
}
