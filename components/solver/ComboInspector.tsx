'use client';

import { useMemo, useState } from 'react';
import {
  buildHandClassBreakdown,
  type FormattedCombo,
  SUIT_COLORS,
  SUIT_NAMES,
  SUIT_SYMBOLS,
} from '@/lib/combo-breakdown';
import type { Card } from '@/lib/cards';
import type { ClassRow } from '@/lib/solver/client';

export interface ComboInspectorProps {
  label: string;
  row?: ClassRow;
  board?: Card[];
  colors: Record<string, string>;
  showMetrics?: boolean;
  className?: string;
}

function formatPercent(fraction: number): string {
  if (fraction <= 0) return '0%';
  if (fraction >= 1) return '100%';
  const p = fraction * 100;
  if (p < 0.1) return '<0.1%';
  if (p > 99.9) return '>99.9%';
  return `${Math.round(p * 10) / 10}%`;
}

function ComboCard({
  rank,
  suit,
  inactive,
}: {
  rank: string;
  suit: number;
  inactive: boolean;
}) {
  const activeSuitColors = [
    'text-emerald-950',
    'text-blue-950',
    'text-red-950',
    'text-zinc-950',
  ];
  return (
    <span
      className={`inline-flex items-center gap-0.5 font-semibold ${inactive ? SUIT_COLORS[suit].text : activeSuitColors[suit]}`}
    >
      {rank}
      <span>{SUIT_SYMBOLS[suit]}</span>
    </span>
  );
}

export function ComboInspector({
  label,
  row,
  board = [],
  colors,
  showMetrics = false,
  className = '',
}: ComboInspectorProps) {
  const breakdown = useMemo(
    () => buildHandClassBreakdown(label, row, board),
    [label, row, board]
  );
  const [suitFilter, setSuitFilter] = useState<number | 'all'>('all');
  const [sortBy, setSortBy] = useState<'default' | 'ev' | 'weight'>('default');
  const filteredCombos = useMemo(() => {
    let list = breakdown.combos;
    if (suitFilter !== 'all') {
      list = list.filter(
        (c) => c.card0Suit === suitFilter || c.card1Suit === suitFilter
      );
    }
    if (sortBy !== 'default') {
      list = [...list].sort((a, b) => {
        if (a.isBlocked !== b.isBlocked) return a.isBlocked ? 1 : -1;
        return sortBy === 'ev' ? b.ev - a.ev : b.weight - a.weight;
      });
    }
    return list;
  }, [breakdown.combos, suitFilter, sortBy]);

  return (
    <aside
      aria-label={`Combination details for ${breakdown.friendlyName}`}
      className={`min-w-0 text-fg ${className}`}
      data-solver-combo-inspector
    >
      <header className="flex flex-wrap items-baseline justify-between gap-2 border-b border-border pb-2">
        <h2 className="flex items-baseline gap-2 text-sm">
          <strong className="font-mono text-lg">{breakdown.label}</strong>
          <span className="text-muted">{breakdown.friendlyName}</span>
        </h2>
        <span className="text-xs text-muted">
          {breakdown.activeCombos.toFixed(1)} / {breakdown.totalCombos} combos
          {breakdown.blockedCombos > 0 &&
            ` · ${breakdown.blockedCombos} blocked`}
        </span>
      </header>
      <div className="flex flex-wrap items-center justify-between gap-x-2 py-1">
        <div
          className="flex items-center"
          role="group"
          aria-label="Filter combinations by suit"
        >
          {(['all', 0, 1, 2, 3] as const).map((suit) => (
            <button
              key={suit}
              type="button"
              aria-label={
                suit === 'all' ? 'All suits' : `Filter by ${SUIT_NAMES[suit]}`
              }
              aria-pressed={suitFilter === suit}
              onClick={() => setSuitFilter(suit)}
              className={`min-h-11 min-w-9 border-b-2 px-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${suitFilter === suit ? 'border-accent text-fg' : 'border-transparent text-muted hover:text-fg'}`}
            >
              {suit === 'all' ? 'All' : SUIT_SYMBOLS[suit]}
            </button>
          ))}
        </div>
        <select
          aria-label="Sort combinations"
          value={sortBy}
          onChange={(event) => setSortBy(event.target.value as typeof sortBy)}
          className="min-h-11 max-w-full bg-transparent text-xs text-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          <option value="default">By suit</option>
          <option value="ev">Highest EV</option>
          <option value="weight">Weight</option>
        </select>
      </div>
      <div
        className={`grid gap-1 ${breakdown.category === 'suited' ? 'grid-cols-2 lg:grid-cols-4' : breakdown.category === 'offsuit' ? 'grid-cols-3 xl:grid-cols-4' : 'grid-cols-3'}`}
        role="list"
        aria-label="Combinations"
      >
        {filteredCombos.map((combo) => (
          <ComboSquare
            key={combo.key}
            combo={combo}
            colors={colors}
            showMetrics={showMetrics}
          />
        ))}
      </div>
      {filteredCombos.length === 0 && (
        <p className="py-4 text-xs text-muted">
          No combinations match this suit.
        </p>
      )}
    </aside>
  );
}

function ComboSquare({
  combo,
  colors,
  showMetrics,
}: {
  combo: FormattedCombo;
  colors: Record<string, string>;
  showMetrics: boolean;
}) {
  const inactive = combo.isBlocked || combo.weight <= 0.001;
  const status = combo.isBlocked
    ? 'Blocked by board'
    : combo.weight <= 0.001
      ? 'Not in range'
      : null;
  const ev = `${combo.ev >= 0 ? '+' : ''}${combo.ev.toFixed(2)}bb`;
  // The fallback is a class average, not a suit-specific solve. Do not present
  // its placeholder equity as a measured combo result.
  const metrics = combo.hasData
    ? `EV ${ev}, equity ${formatPercent(combo.equity)}, weight ${formatPercent(combo.weight)}`
    : 'Class average';
  const description = `${combo.card0Rank} of ${SUIT_NAMES[combo.card0Suit]}, ${combo.card1Rank} of ${SUIT_NAMES[combo.card1Suit]}: ${status ?? `${combo.actions.map((a) => `${a.action} ${formatPercent(a.freq)}`).join(', ')}; ${metrics}`}`;

  return (
    <div
      role="listitem"
      tabIndex={0}
      aria-label={description}
      title={description}
      data-solver-combo-row
      data-combo-key={combo.key}
      className={`relative flex aspect-square min-w-0 flex-col bg-surface-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${inactive ? 'opacity-60' : ''}`}
    >
      {!inactive && (
        <div
          className="pointer-events-none absolute inset-0 flex"
          aria-hidden="true"
        >
          {combo.actions.map((action) => (
            <span
              key={action.action}
              style={{
                width: `${action.freq * 100}%`,
                background: colors[action.action],
              }}
            />
          ))}
        </div>
      )}
      <div className="relative flex flex-1 flex-col p-1.5 sm:p-2">
        <div className="flex w-fit flex-wrap items-center gap-1.5 text-xl leading-tight sm:text-2xl">
          <ComboCard
            rank={combo.card0Rank}
            suit={combo.card0Suit}
            inactive={inactive}
          />
          <ComboCard
            rank={combo.card1Rank}
            suit={combo.card1Suit}
            inactive={inactive}
          />
        </div>
        {status ? (
          <span className="mt-auto text-[11px] text-muted">{status}</span>
        ) : (
          <div className="mt-auto py-0.5 text-[11px] font-medium leading-snug text-black/90 sm:text-sm">
            {combo.actions.map((action) => (
              <div key={action.action} className="flex justify-between gap-1">
                <span className="truncate">
                  {action.action.replaceAll('%', '')}
                </span>
                <span className="tabular-nums">
                  {formatPercent(action.freq).replace('%', '')}
                </span>
              </div>
            ))}
            {showMetrics && (
              <div className="mt-1 border-t border-black/20 pt-1 text-black/70">
                {combo.hasData ? (
                  <>
                    <span className="block">EV {ev}</span>
                    <span>
                      EQ {formatPercent(combo.equity).replace('%', '')} · Wt{' '}
                      {formatPercent(combo.weight).replace('%', '')}
                    </span>
                  </>
                ) : (
                  'Class average'
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
