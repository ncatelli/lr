use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
pub(crate) struct OrderedSet<T: Hash> {
    elem_idx: HashMap<u64, usize>,
    elems: Vec<T>,
}

impl<T: Hash> OrderedSet<T> {
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

        if !self.elem_idx.contains_key(&elem_hash) {
            self.elem_idx.insert(elem_hash, slot);
            self.elems.push(elem);

            true
        } else {
            false
        }
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
    pub fn contains(&self, elem: &T) -> bool {
        self.position(elem).is_some()
    }

    pub fn to_vec(self) -> Vec<T> {
        self.elems
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
        value.to_vec()
    }
}

impl<T: Hash> Default for OrderedSet<T> {
    fn default() -> Self {
        Self {
            elem_idx: Default::default(),
            elems: Default::default(),
        }
    }
}
