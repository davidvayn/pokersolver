'use client';

import { useEffect, useRef, useState } from 'react';
import {
  CARD_SUIT_GLYPHS,
  PokerCard,
} from '@/components/cards/PokerCard';
import { Card, RANKS, makeCard, cardToStr } from '@/lib/cards';

const SUIT_SYMBOL = CARD_SUIT_GLYPHS;

export function PlayingCard({
  card,
  size = 'md',
  onClick,
  dimmed,
}: {
  card: Card;
  size?: 'sm' | 'md' | 'lg';
  onClick?: (e: React.MouseEvent<HTMLButtonElement>) => void;
  dimmed?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={!onClick}
      className={
        'block rounded-md ' +
        (dimmed ? 'opacity-30 ' : '') +
        (onClick
          ? 'hover:ring-2 hover:ring-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent '
          : '')
      }
    >
      <PokerCard card={card} size={size} />
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
                aria-label={cardToStr(card)}
                className={
                  'flex h-8 w-7 items-center justify-center rounded border text-xs font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ' +
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
  size = 'md',
}: {
  count: number;
  cards: Card[];
  used: Set<Card>;
  onChange: (cards: Card[]) => void;
  label?: string;
  size?: 'sm' | 'md' | 'lg';
}) {
  const addDims =
    size === 'lg' ? 'h-16 w-12 text-lg' : size === 'sm' ? 'h-7 w-5' : 'h-10 w-7';
  const [open, setOpen] = useState(false);
  const [slot, setSlot] = useState(0);
  const popoverRef = useRef<HTMLDivElement>(null);
  const openerRef = useRef<HTMLButtonElement | null>(null);

  function openPicker(i: number, el: HTMLButtonElement) {
    setSlot(i);
    openerRef.current = el;
    setOpen(true);
  }

  function closePicker() {
    setOpen(false);
    openerRef.current?.focus();
  }

  // While open: Escape closes and focus moves into the picker.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        closePicker();
      }
    };
    window.addEventListener('keydown', onKey);
    popoverRef.current?.focus();
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);

  function pick(c: Card) {
    const next = cards.slice();
    next[slot] = c;
    onChange(next.filter((x) => x !== undefined));
    closePicker();
  }

  function removeAt(i: number) {
    const next = cards.slice();
    next.splice(i, 1);
    onChange(next);
  }

  // Exclude cards already placed in this component's OTHER slots (the parent
  // strips this side's own hand from `used`), while keeping the card currently
  // being edited selectable.
  const usedExceptEditing = new Set(used);
  cards.forEach((c, i) => {
    if (c === undefined) return;
    if (i === slot) usedExceptEditing.delete(c);
    else usedExceptEditing.add(c);
  });

  return (
    <div className="relative">
      {label && <div className="mb-1 text-xs text-muted">{label}</div>}
      <div className="flex items-center gap-1">
        {Array.from({ length: count }).map((_, i) => {
          const c = cards[i];
          return c !== undefined ? (
            <div key={i} className="group relative">
              <PlayingCard
                card={c}
                size={size}
                onClick={(e) => openPicker(i, e.currentTarget)}
              />
              <button
                onClick={() => removeAt(i)}
                className="absolute -right-1 -top-1 flex h-4 w-4 items-center justify-center rounded-full bg-raise text-[10px] text-white opacity-0 transition-opacity focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent group-hover:opacity-100"
                aria-label={`Remove ${cardToStr(c)}`}
              >
                ×
              </button>
            </div>
          ) : (
            <button
              key={i}
              onClick={(e) => openPicker(i, e.currentTarget)}
              aria-label="Add card"
              className={
                'flex items-center justify-center rounded-md border border-dashed border-border text-muted hover:border-accent hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ' +
                addDims
              }
            >
              +
            </button>
          );
        })}
      </div>
      {open && (
        <>
          {/* Click-outside backdrop closes the picker (Escape also closes) */}
          <div
            className="fixed inset-0 z-40"
            onClick={closePicker}
            aria-hidden
          />
          <div
            ref={popoverRef}
            role="dialog"
            aria-label="Pick a card"
            tabIndex={-1}
            className="absolute left-0 top-full z-50 mt-2 rounded-lg border-2 border-accent bg-surface p-3 shadow-card outline-none ring-1 ring-black/20"
          >
            <div className="mb-2 flex items-center justify-between text-xs text-muted">
              <span>Pick a card</span>
              <button
                onClick={closePicker}
                className="rounded hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
              >
                Close
              </button>
            </div>
            <CardGrid used={usedExceptEditing} onPick={pick} />
          </div>
        </>
      )}
    </div>
  );
}
