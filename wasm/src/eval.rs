//! Compact 5-to-7 card hand evaluator. Port of the TypeScript evaluator in
//! `lib/evaluator.ts`; higher score = better hand.

#[inline]
fn card_rank(c: u8) -> u32 {
    (c >> 2) as u32
}
#[inline]
fn card_suit(c: u8) -> usize {
    (c & 3) as usize
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

/// Evaluate the best 5-card hand from 5..=7 cards.
pub fn evaluate(cards: &[u8]) -> u32 {
    let mut rank_count = [0u8; 13];
    let mut suit_rank_mask = [0u32; 4];
    let mut rank_mask = 0u32;

    for &c in cards {
        let r = card_rank(c);
        let s = card_suit(c);
        rank_count[r as usize] += 1;
        suit_rank_mask[s] |= 1 << r;
        rank_mask |= 1 << r;
    }

    // Flush / straight flush
    for s in 0..4 {
        let mask = suit_rank_mask[s];
        if mask.count_ones() >= 5 {
            let sf = straight_high(mask);
            if sf >= 0 {
                return score(STRAIGHT_FLUSH, sf as u32);
            }
            return score(FLUSH, top_n(mask, 5));
        }
    }

    let mut quad: i32 = -1;
    let mut trips: Vec<u32> = Vec::new();
    let mut pairs: Vec<u32> = Vec::new();
    for r in (0..13u32).rev() {
        match rank_count[r as usize] {
            4 => quad = r as i32,
            3 => trips.push(r),
            2 => pairs.push(r),
            _ => {}
        }
    }

    if quad >= 0 {
        let kicker = highest_except(rank_mask, &[quad as u32]);
        return score(FOUR_KIND, ((quad as u32) << 4) | kicker);
    }

    if trips.len() >= 1 && (trips.len() >= 2 || pairs.len() >= 1) {
        let three = trips[0];
        let pair = if trips.len() >= 2 { trips[1] } else { pairs[0] };
        return score(FULL_HOUSE, (three << 4) | pair);
    }

    let st = straight_high(rank_mask);
    if st >= 0 {
        return score(STRAIGHT, st as u32);
    }

    if trips.len() >= 1 {
        let three = trips[0];
        return score(THREE_KIND, (three << 16) | top_n_except(rank_mask, 2, three));
    }

    if pairs.len() >= 2 {
        let hi = pairs[0];
        let lo = pairs[1];
        let kicker = highest_except(rank_mask, &[hi, lo]);
        return score(TWO_PAIR, (hi << 8) | (lo << 4) | kicker);
    }

    if pairs.len() == 1 {
        let p = pairs[0];
        return score(ONE_PAIR, (p << 16) | top_n_except(rank_mask, 3, p));
    }

    score(HIGH_CARD, top_n(rank_mask, 5))
}

fn straight_high(mask: u32) -> i32 {
    let mut hi = 12i32;
    while hi >= 4 {
        let need = 0b11111u32 << (hi - 4);
        if (mask & need) == need {
            return hi;
        }
        hi -= 1;
    }
    let wheel = (1 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3);
    if (mask & wheel) == wheel {
        return 3;
    }
    -1
}

fn top_n(mask: u32, n: usize) -> u32 {
    let mut out = 0u32;
    let mut count = 0;
    let mut r = 12i32;
    while r >= 0 && count < n {
        if mask & (1 << r) != 0 {
            out = (out << 4) | r as u32;
            count += 1;
        }
        r -= 1;
    }
    out
}

fn top_n_except(mask: u32, n: usize, except: u32) -> u32 {
    top_n(mask & !(1 << except), n)
}

fn highest_except(mask: u32, except: &[u32]) -> u32 {
    let mut m = mask;
    for &e in except {
        m &= !(1 << e);
    }
    for r in (0..13u32).rev() {
        if m & (1 << r) != 0 {
            return r;
        }
    }
    0
}
