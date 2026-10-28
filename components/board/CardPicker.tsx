'use client';

import { useEffect, useRef, useState, type CSSProperties } from 'react';
import { createPortal } from 'react-dom';
import { CARD_SUIT_GLYPHS, PokerCard } from '@/components/cards/PokerCard';
import { Card, RANKS, makeCard, cardToStr } from '@/lib/cards';

const SUIT_SYMBOL = CARD_SUIT_GLYPHS;

export function PlayingCard({
  card,
  size = 'md',
  onClick,
  dimmed,
}: {
  card: Card;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  onClick?: (e: React.MouseEvent<HTMLButtonElement>) => void;
  dimmed?: boolean;
}) {
  return (
    <button
      type="button"
      aria-label={onClick ? `Edit ${cardToStr(card)}` : undefined}
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
    <div className="card-picker-grid">
      {[3, 2, 1, 0].map((suit) => (
        <div key={suit} className="card-picker-suit">
          {Array.from({ length: 13 }).map((_, i) => {
            const rank = 12 - i;
            const card = makeCard(rank, suit);
            const isUsed = used.has(card);
            return (
              <button
                key={card}
                type="button"
                disabled={isUsed}
                onClick={() => onPick(card)}
                aria-label={cardToStr(card)}
                className={
                  'card-picker-option flex min-w-0 items-center justify-center rounded border font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ' +
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
  size?: 'sm' | 'md' | 'lg' | 'xl';
}) {
  const addDims =
    size === 'xl'
      ? 'h-[72px] w-[52px] text-xl'
      : size === 'lg'
        ? 'h-16 w-12 text-lg'
        : size === 'sm'
          ? 'h-7 w-5'
          : 'h-10 w-7';
  const [open, setOpen] = useState(false);
  const [slot, setSlot] = useState(0);
  const popoverRef = useRef<HTMLDivElement>(null);
  const slotsRef = useRef<HTMLDivElement>(null);
  const openerRef = useRef<HTMLButtonElement | null>(null);
  const [position, setPosition] = useState<CSSProperties>({
    left: 16,
    top: 16,
  });

  function openPicker(i: number, el: HTMLButtonElement) {
    setSlot(i);
    openerRef.current = el;
    setOpen(true);
  }

  function closePicker() {
    setOpen(false);
    requestAnimationFrame(() => {
      slotsRef.current
        ?.querySelector<HTMLButtonElement>(
          `[data-card-slot="${slot}"] > button`
        )
        ?.focus({ preventScroll: true });
    });
  }

  // While open: Escape closes and focus moves into the picker.
  useEffect(() => {
    if (!open) return;
    function place() {
      const box = openerRef.current?.getBoundingClientRect();
      const popup = popoverRef.current;
      if (!box || !popup) return;
      const width = Math.min(680, window.innerWidth - 32);
      setPosition({
        width,
        left: Math.max(16, Math.min(box.left, window.innerWidth - width - 16)),
        top: Math.max(
          16,
          Math.min(box.bottom + 8, window.innerHeight - popup.offsetHeight - 16)
        ),
      });
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        closePicker();
      }
      if (e.key === 'Tab') {
        const buttons = Array.from(
          popoverRef.current?.querySelectorAll<HTMLButtonElement>(
            'button:not(:disabled)'
          ) ?? []
        );
        const first = buttons[0];
        const last = buttons.at(-1);
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last?.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first?.focus();
        }
      }
    };
    place();
    window.addEventListener('keydown', onKey);
    window.addEventListener('resize', place);
    window.addEventListener('scroll', place, true);
    popoverRef.current
      ?.querySelector<HTMLButtonElement>('.card-picker-option:not(:disabled)')
      ?.focus({ preventScroll: true });
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('resize', place);
      window.removeEventListener('scroll', place, true);
    };
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
    <div ref={slotsRef} className="relative">
      {label && <div className="mb-1 text-xs text-muted">{label}</div>}
      <div className="card-slots flex items-center gap-1">
        {Array.from({ length: count }).map((_, i) => {
          const c = cards[i];
          return (
            <div key={i} data-card-slot={i} className="group relative">
              {c !== undefined ? (
                <>
                  <PlayingCard
                    card={c}
                    size={size}
                    onClick={(e) => openPicker(i, e.currentTarget)}
                  />
                  <button
                    type="button"
                    onClick={() => removeAt(i)}
                    className="absolute -right-1 -top-1 flex h-6 w-6 items-center justify-center rounded-full bg-raise text-sm text-white opacity-0 transition-opacity focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent group-hover:opacity-100"
                    aria-label={`Remove ${cardToStr(c)}`}
                  >
                    ×
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  onClick={(e) => openPicker(i, e.currentTarget)}
                  aria-label={`Add card ${i + 1}`}
                  className={
                    'flex items-center justify-center rounded-md border border-dashed border-border text-muted hover:border-accent hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ' +
                    addDims
                  }
                >
                  +
                </button>
              )}
            </div>
          );
        })}
      </div>
      {open &&
        createPortal(
          <>
            {/* Click-outside backdrop closes the picker (Escape also closes) */}
            <div
              className="fixed inset-0 z-[190] bg-black/15"
              onClick={closePicker}
              aria-hidden
            />
            <div
              ref={popoverRef}
              role="dialog"
              aria-modal="true"
              aria-label="Pick a card"
              tabIndex={-1}
              style={position}
              className="card-picker-popover"
            >
              <div className="mb-3 flex items-center justify-between text-sm font-medium text-muted">
                <span>Pick a card</span>
                <button
                  type="button"
                  onClick={closePicker}
                  className="min-h-11 rounded px-2 hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                >
                  Close
                </button>
              </div>
              <CardGrid used={usedExceptEditing} onPick={pick} />
            </div>
          </>,
          document.body
        )}
    </div>
  );
}
