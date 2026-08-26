//! Compact 5-to-7 card evaluator adapted from `wasm/src/eval.rs`.
//! Higher scores represent stronger hands.

#[inline]
fn card_rank(card: u8) -> u32 {
    (card >> 2) as u32
}

#[inline]
fn card_suit(card: u8) -> usize {
    (card & 3) as usize
}

const STRAIGHT_FLUSH: u32 = 8;
const FOUR_KIND: u32 = 7;
const FULL_HOUSE: u32 = 6;
const FLUSH: u32 = 5;
const STRAIGHT: u32 = 4;
const THREE_KIND: u32 = 3;
const TWO_PAIR: u32 = 2;
const ONE_PAIR: u32 = 1;
const HIGH_CARD: u32 = 0;

#[inline]
fn score(category: u32, tiebreak: u32) -> u32 {
    (category << 24) | tiebreak
}

pub fn evaluate(cards: &[u8]) -> u32 {
    assert!((5..=7).contains(&cards.len()));
    let mut rank_count = [0u8; 13];
    let mut suit_rank_mask = [0u32; 4];
    let mut rank_mask = 0u32;

    for &card in cards {
        let rank = card_rank(card);
        let suit = card_suit(card);
        rank_count[rank as usize] += 1;
        suit_rank_mask[suit] |= 1 << rank;
        rank_mask |= 1 << rank;
    }

    for mask in suit_rank_mask {
        if mask.count_ones() >= 5 {
            let straight_flush = straight_high(mask);
            if straight_flush >= 0 {
                return score(STRAIGHT_FLUSH, straight_flush as u32);
            }
            return score(FLUSH, top_n(mask, 5));
        }
    }

    let mut quad = -1i32;
    // Seven cards can contain at most two trip ranks or three pair ranks.
    // Keep these tiny collections on the stack: this evaluator is called
    // millions of times while building postflop information-set buckets.
    let mut trips = [0u32; 2];
    let mut trip_count = 0usize;
    let mut pairs = [0u32; 3];
    let mut pair_count = 0usize;
    for rank in (0..13u32).rev() {
        match rank_count[rank as usize] {
            4 => quad = rank as i32,
            3 => {
                trips[trip_count] = rank;
                trip_count += 1;
            }
            2 => {
                pairs[pair_count] = rank;
                pair_count += 1;
            }
            _ => {}
        }
    }

    if quad >= 0 {
        let kicker = highest_except(rank_mask, &[quad as u32]);
        return score(FOUR_KIND, ((quad as u32) << 4) | kicker);
    }
    if trip_count > 0 && (trip_count >= 2 || pair_count > 0) {
        let pair = if trip_count >= 2 { trips[1] } else { pairs[0] };
        return score(FULL_HOUSE, (trips[0] << 4) | pair);
    }
    let straight = straight_high(rank_mask);
    if straight >= 0 {
        return score(STRAIGHT, straight as u32);
    }
    if trip_count > 0 {
        return score(
            THREE_KIND,
            (trips[0] << 16) | top_n_except(rank_mask, 2, trips[0]),
        );
    }
    if pair_count >= 2 {
        let kicker = highest_except(rank_mask, &[pairs[0], pairs[1]]);
        return score(TWO_PAIR, (pairs[0] << 8) | (pairs[1] << 4) | kicker);
    }
    if pair_count == 1 {
        return score(
            ONE_PAIR,
            (pairs[0] << 16) | top_n_except(rank_mask, 3, pairs[0]),
        );
    }
    score(HIGH_CARD, top_n(rank_mask, 5))
}

fn straight_high(mask: u32) -> i32 {
    for high in (4..=12i32).rev() {
        let needed = 0b11111u32 << (high - 4);
        if (mask & needed) == needed {
            return high;
        }
    }
    let wheel = (1 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3);
    if (mask & wheel) == wheel {
        return 3;
    }
    -1
}

fn top_n(mask: u32, count: usize) -> u32 {
    let mut result = 0u32;
    let mut found = 0;
    for rank in (0..13u32).rev() {
        if mask & (1 << rank) != 0 {
            result = (result << 4) | rank;
            found += 1;
            if found == count {
                break;
            }
        }
    }
    result
}

fn top_n_except(mask: u32, count: usize, except: u32) -> u32 {
    top_n(mask & !(1 << except), count)
}

fn highest_except(mask: u32, except: &[u32]) -> u32 {
    let mut available = mask;
    for &rank in except {
        available &= !(1 << rank);
    }
    for rank in (0..13u32).rev() {
        if available & (1 << rank) != 0 {
            return rank;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_known_seven_card_hands() {
        let straight_flush = evaluate(&[51, 47, 43, 39, 35, 0, 1]);
        let quads = evaluate(&[48, 49, 50, 51, 47, 0, 1]);
        let full_house = evaluate(&[48, 49, 50, 44, 45, 0, 1]);
        assert!(straight_flush > quads);
        assert!(quads > full_house);
    }

    #[test]
    fn wheel_is_lower_than_six_high_straight() {
        let wheel = evaluate(&[48, 0, 5, 10, 15]);
        let six_high = evaluate(&[0, 5, 10, 15, 16]);
        assert!(six_high > wheel);
    }

    #[test]
    fn seven_card_rank_buffers_cover_two_trips_and_three_pairs() {
        let two_trips = evaluate(&[48, 49, 50, 44, 45, 46, 40]);
        assert_eq!(two_trips, score(FULL_HOUSE, (12 << 4) | 11));

        let three_pairs = evaluate(&[48, 49, 44, 45, 40, 41, 36]);
        assert_eq!(three_pairs, score(TWO_PAIR, (12 << 8) | (11 << 4) | 10));
    }
}
