'use client';

import { useState } from 'react';
import { Card, RANKS, SUITS, makeCard, cardToStr, cardRank, cardSuit } from '@/lib/cards';

const SUIT_SYMBOL = ['♣', '♦', '♥', '♠'];
const SUIT_COLOR = [
  'text-emerald-500',
  'text-sky-400',
  'text-rose-400',
  'text-fg',
];

export function PlayingCard({
  card,
  size = 'md',
  onClick,
  dimmed,
}: {
  card: Card;
  size?: 'sm' | 'md' | 'lg';
  onClick?: () => void;
  dimmed?: boolean;
}) {
  const r = cardRank(card);
  const s = cardSuit(card);
  const dims =
    size === 'lg'
      ? 'h-14 w-10 text-lg'
      : size === 'sm'
        ? 'h-7 w-5 text-[11px]'
        : 'h-10 w-7 text-sm';
  return (
    <button
      onClick={onClick}
      disabled={!onClick}
      className={
        `flex flex-col items-center justify-center rounded-md border border-border bg-white font-bold shadow-sm ${dims} ` +
        (dimmed ? 'opacity-30 ' : '') +
        (onClick ? 'hover:ring-2 hover:ring-accent ' : '')
      }
    >
      <span className={SUIT_COLOR[s] === 'text-fg' ? 'text-gray-900' : SUIT_COLOR[s]}>
        {RANKS[r]}
      </span>
      <span className={SUIT_COLOR[s] === 'text-fg' ? 'text-gray-900' : SUIT_COLOR[s]}>
        {SUIT_SYMBOL[s]}
      </span>
    </button>
  );
}

/** A grid of all 52 cards; used cards are disabled. */
export function CardGrid({
  used,
  onPick,
}: {
  used: Set<Card>;
  onPick: (c: Card) => void;
}) {
  return (
    <div className="flex flex-col gap-1">
      {[3, 2, 1, 0].map((suit) => (
        <div key={suit} className="flex gap-1">
          {Array.from({ length: 13 }).map((_, i) => {
            const rank = 12 - i;
            const card = makeCard(rank, suit);
            const isUsed = used.has(card);
            return (
              <button
                key={card}
                disabled={isUsed}
                onClick={() => onPick(card)}
                className={
                  'flex h-8 w-7 items-center justify-center rounded border text-xs font-semibold transition-colors ' +
                  (isUsed
                    ? 'cursor-not-allowed border-border bg-surface-2 text-muted opacity-40'
                    : 'border-border bg-white text-gray-900 hover:ring-2 hover:ring-accent')
                }
              >
                <span
                  className={
                    suit === 0
                      ? 'text-emerald-600'
                      : suit === 1
                        ? 'text-sky-600'
                        : suit === 2
                          ? 'text-rose-600'
                          : 'text-gray-900'
                  }
                >
                  {RANKS[rank]}
                  {SUIT_SYMBOL[suit]}
                </span>
              </button>
            );
          })}
        </div>
      ))}
    </div>
  );
}

/** Slots that open a picker; used for hole cards (n=2) or board (n=5). */
export function CardSlots({
  count,
  cards,
  used,
  onChange,
  label,
}: {
  count: number;
  cards: Card[];
  used: Set<Card>;
  onChange: (cards: Card[]) => void;
  label?: string;
}) {
  const [open, setOpen] = useState(false);
  const [slot, setSlot] = useState(0);

  function pick(c: Card) {
    const next = cards.slice();
    next[slot] = c;
    onChange(next.filter((x) => x !== undefined));
    setOpen(false);
  }

  function removeAt(i: number) {
    const next = cards.slice();
    next.splice(i, 1);
    onChange(next);
  }

  const usedExceptEditing = new Set(used);
  cards.forEach((c, i) => {
    if (i === slot) usedExceptEditing.delete(c);
  });

  return (
    <div className="relative">
      {label && <div className="mb-1 text-xs text-muted">{label}</div>}
      <div className="flex items-center gap-1">
        {Array.from({ length: count }).map((_, i) => {
          const c = cards[i];
          return c !== undefined ? (
            <div key={i} className="group relative">
              <PlayingCard card={c} onClick={() => { setSlot(i); setOpen(true); }} />
              <button
                onClick={() => removeAt(i)}
                className="absolute -right-1 -top-1 hidden h-4 w-4 items-center justify-center rounded-full bg-raise text-[10px] text-white group-hover:flex"
                aria-label="remove"
              >
                ×
              </button>
            </div>
          ) : (
            <button
              key={i}
              onClick={() => { setSlot(i); setOpen(true); }}
              className="flex h-10 w-7 items-center justify-center rounded-md border border-dashed border-border text-muted hover:border-accent hover:text-accent"
            >
              +
            </button>
          );
        })}
      </div>
      {open && (
        <div className="absolute left-0 top-full z-50 mt-2 rounded-lg border border-border bg-surface p-3 shadow-card">
          <div className="mb-2 flex items-center justify-between text-xs text-muted">
            <span>Pick a card</span>
            <button onClick={() => setOpen(false)} className="hover:text-fg">
              Close
            </button>
          </div>
          <CardGrid used={usedExceptEditing} onPick={pick} />
        </div>
      )}
    </div>
  );
}
