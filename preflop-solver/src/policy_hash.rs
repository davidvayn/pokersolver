//! Canonical serving-state hash shared with `lib/practice-engine.ts`.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CanonicalPolicyState {
    pub model_version: String,
    pub depth_bb: f64,
    pub actor: String,
    pub street: String,
    pub hole_cards: [u8; 2],
    pub board: Vec<u8>,
    pub pot_bb: f64,
    pub button_small_blind_bet_bb: f64,
    pub big_blind_bet_bb: f64,
    pub button_small_blind_stack_bb: f64,
    pub big_blind_stack_bb: f64,
    pub public_history: Vec<String>,
}

impl CanonicalPolicyState {
    pub fn canonical_string(&self) -> String {
        let mut holes = self.hole_cards;
        holes.sort_unstable();
        let mut board = self.board.clone();
        board.sort_unstable();
        let holes = holes
            .iter()
            .map(u8::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let board = board
            .iter()
            .map(u8::to_string)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "hu-cash-v1|{}|{:.3}|{}|{}|{}|{}|{:.3}|{:.3}|{:.3}|{:.3}|{:.3}|{}",
            self.model_version,
            self.depth_bb,
            self.actor,
            self.street,
            holes,
            board,
            self.pot_bb,
            self.button_small_blind_bet_bb,
            self.big_blind_bet_bb,
            self.button_small_blind_stack_bb,
            self.big_blind_stack_bb,
            self.public_history.join("/"),
        )
    }

    pub fn sha256(&self) -> String {
        let digest = Sha256::digest(self.canonical_string().as_bytes());
        digest.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_typescript_serving_hash_fixture() {
        let state = CanonicalPolicyState {
            model_version: "test-v1".to_owned(),
            depth_bb: 20.0,
            actor: "button-small-blind".to_owned(),
            street: "preflop".to_owned(),
            hole_cards: [51, 47],
            board: Vec::new(),
            pot_bb: 0.0,
            button_small_blind_bet_bb: 0.5,
            big_blind_bet_bb: 1.0,
            button_small_blind_stack_bb: 19.5,
            big_blind_stack_bb: 19.0,
            public_history: Vec::new(),
        };
        assert_eq!(
            state.sha256(),
            "b61126532572af5ab17edbac4fc4a5a9976be22cec96813bff5c2fce64202ccb"
        );
    }
}
