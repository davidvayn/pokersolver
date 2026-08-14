use serde::{Deserialize, Serialize};

pub const RANKS: &[u8; 13] = b"23456789TJQKA";
pub const SUITS: &[u8; 4] = b"cdhs";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Combo {
    pub high: u8,
    pub low: u8,
}

impl Combo {
    pub fn new(a: u8, b: u8) -> Self {
        assert!(a < 52 && b < 52 && a != b);
        if a > b {
            Self { high: a, low: b }
        } else {
            Self { high: b, low: a }
        }
    }

    pub fn key(self) -> usize {
        (self.high as usize * (self.high as usize - 1)) / 2 + self.low as usize
    }

    pub fn cards(self) -> [u8; 2] {
        [self.high, self.low]
    }

    pub fn overlaps(self, other: Self) -> bool {
        self.high == other.high
            || self.high == other.low
            || self.low == other.high
            || self.low == other.low
    }

    pub fn label(self) -> String {
        let high_rank = self.high >> 2;
        let low_rank = self.low >> 2;
        if high_rank == low_rank {
            let rank = RANKS[high_rank as usize] as char;
            return format!("{rank}{rank}");
        }

        let (first, second) = if high_rank > low_rank {
            (high_rank, low_rank)
        } else {
            (low_rank, high_rank)
        };
        let suited = (self.high & 3) == (self.low & 3);
        format!(
            "{}{}{}",
            RANKS[first as usize] as char,
            RANKS[second as usize] as char,
            if suited { 's' } else { 'o' }
        )
    }

    pub fn card_strings(self) -> [String; 2] {
        [card_to_string(self.high), card_to_string(self.low)]
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ComboIdentity {
    pub combo_key: usize,
    pub cards: [u8; 2],
    pub card_names: [String; 2],
    pub label: String,
}

impl From<Combo> for ComboIdentity {
    fn from(combo: Combo) -> Self {
        Self {
            combo_key: combo.key(),
            cards: combo.cards(),
            card_names: combo.card_strings(),
            label: combo.label(),
        }
    }
}

pub fn all_combos() -> Vec<Combo> {
    let mut combos = Vec::with_capacity(1326);
    for high in 1..52 {
        for low in 0..high {
            combos.push(Combo::new(high, low));
        }
    }
    combos
}

pub fn card_to_string(card: u8) -> String {
    let rank = RANKS[(card >> 2) as usize] as char;
    let suit = SUITS[(card & 3) as usize] as char;
    format!("{rank}{suit}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combo_keys_match_canonical_triangular_encoding() {
        let combos = all_combos();
        assert_eq!(combos.len(), 1326);
        for (expected, combo) in combos.into_iter().enumerate() {
            assert_eq!(combo.key(), expected);
        }
    }

    #[test]
    fn labels_preserve_pair_and_suitedness() {
        assert_eq!(Combo::new(51, 50).label(), "AA");
        assert_eq!(Combo::new(51, 47).label(), "AKs");
        assert_eq!(Combo::new(51, 46).label(), "AKo");
        assert_eq!(Combo::new(51, 46).card_strings(), ["As", "Kh"]);
    }
}
