'use client';

import { useMemo, useState } from 'react';
import { X, ShieldAlert, Check } from 'lucide-react';
import {
  buildHandClassBreakdown,
  FormattedCombo,
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
  onClose: () => void;
  className?: string;
}

function formatPercent(fraction: number): string {
  if (fraction <= 0) return '0%';
  if (fraction >= 1) return '100%';
  const p = fraction * 100;
  if (p < 0.1) return '<0.1%';
  if (p > 99.9) return '>99.9%';
  const rounded = Math.round(p * 10) / 10;
  return Number.isInteger(rounded)
    ? `${rounded.toFixed(0)}%`
    : `${rounded.toFixed(1)}%`;
}

function CardBadge({
  rank,
  suit,
  isBlocked = false,
}: {
  rank: string;
  suit: number;
  isBlocked?: boolean;
}) {
  const suitColor = SUIT_COLORS[suit] ?? SUIT_COLORS[3];
  const symbol = SUIT_SYMBOLS[suit] ?? '♠';

  return (
    <span
      className={`inline-flex items-center gap-0.5 rounded px-1.5 py-0.5 font-mono text-xs font-bold leading-none border shadow-xs transition-opacity ${
        isBlocked
          ? 'opacity-40 line-through bg-surface-2 text-muted border-border'
          : `${suitColor.bg} ${suitColor.text} ${suitColor.border}`
      }`}
    >
      <span>{rank}</span>
      <span className="text-[11px] leading-none" aria-hidden="true">
        {symbol}
      </span>
    </span>
  );
}

export function ComboInspector({
  label,
  row,
  board = [],
  colors,
  onClose,
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

    if (sortBy === 'ev') {
      list = [...list].sort((a, b) => {
        if (a.isBlocked !== b.isBlocked) return a.isBlocked ? 1 : -1;
        return b.ev - a.ev;
      });
    } else if (sortBy === 'weight') {
      list = [...list].sort((a, b) => {
        if (a.isBlocked !== b.isBlocked) return a.isBlocked ? 1 : -1;
        return b.weight - a.weight;
      });
    }

    return list;
  }, [breakdown.combos, suitFilter, sortBy]);

  return (
    <aside
      aria-label={`Combination details for ${breakdown.friendlyName}`}
      className={`flex min-h-0 flex-col overflow-hidden rounded-lg border border-border bg-surface text-fg shadow-sm ${className}`}
      data-solver-combo-inspector
    >
      {/* Header */}
      <header className="flex shrink-0 items-center justify-between border-b border-border bg-surface-2/60 px-3 py-2.5">
        <div className="flex min-w-0 items-center gap-2">
          <span className="rounded bg-accent/15 px-2 py-0.5 font-mono text-sm font-bold text-accent ring-1 ring-accent/30">
            {breakdown.label}
          </span>
          <div className="min-w-0">
            <h2 className="truncate text-xs font-semibold leading-tight text-fg">
              {breakdown.friendlyName}
            </h2>
            <div className="flex items-center gap-1.5 text-[11px] text-muted">
              <span className="capitalize">{breakdown.categoryLabel}</span>
              <span>·</span>
              <span className="font-medium text-fg">
                {breakdown.activeCombos.toFixed(1)} / {breakdown.totalCombos} combos
              </span>
              {breakdown.blockedCombos > 0 && (
                <span className="text-raise/90">
                  ({breakdown.blockedCombos} blocked)
                </span>
              )}
            </div>
          </div>
        </div>

        <button
          type="button"
          onClick={onClose}
          aria-label="Close combination details"
          title="Close details (Esc)"
          className="grid h-7 w-7 place-items-center rounded-md text-muted transition-colors hover:bg-surface-2 hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          <X className="h-4 w-4" aria-hidden="true" />
        </button>
      </header>

      {/* Aggregate Class Strategy Summary Bar */}
      {breakdown.classActions.length > 0 && (
        <div className="shrink-0 border-b border-border bg-surface px-3 py-2">
          <div className="mb-1.5 flex items-center justify-between text-[11px]">
            <span className="font-medium text-muted">Aggregate class strategy</span>
            {breakdown.classEv !== undefined && (
              <span className="font-mono font-semibold text-fg">
                EV: <strong className="text-accent">{breakdown.classEv.toFixed(2)}bb</strong>
              </span>
            )}
          </div>
          <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-surface-2">
            {breakdown.classActions.map((action) => {
              if (action.freq < 0.005) return null;
              return (
                <div
                  key={action.action}
                  style={{
                    width: `${action.freq * 100}%`,
                    backgroundColor: colors[action.action] || 'rgb(var(--muted))',
                  }}
                  title={`${action.action}: ${formatPercent(action.freq)}`}
                />
              );
            })}
          </div>
          <div className="mt-1 flex flex-wrap gap-x-2.5 gap-y-1 text-[10px] text-muted">
            {breakdown.classActions.map((action) => (
              <span key={action.action} className="inline-flex items-center gap-1">
                <span
                  className="h-1.5 w-1.5 rounded-full"
                  style={{
                    backgroundColor:
                      colors[action.action] || 'rgb(var(--muted))',
                  }}
                  aria-hidden="true"
                />
                <span>{action.action}</span>
                <span className="font-mono font-semibold text-fg">
                  {formatPercent(action.freq)}
                </span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Filter / Sort bar for offsuit hands with 12 combos */}
      {breakdown.totalCombos > 4 && (
        <div className="flex shrink-0 items-center justify-between border-b border-border bg-surface-2/40 px-3 py-1.5 text-[11px]">
          <div className="flex items-center gap-1">
            <span className="text-[10px] uppercase text-muted font-semibold mr-1">
              Suit:
            </span>
            <button
              type="button"
              onClick={() => setSuitFilter('all')}
              className={`rounded px-1.5 py-0.5 text-[10px] font-medium transition-colors ${
                suitFilter === 'all'
                  ? 'bg-surface text-fg shadow-xs border border-border'
                  : 'text-muted hover:text-fg'
              }`}
            >
              All
            </button>
            {SUIT_SYMBOLS.map((symbol, idx) => (
              <button
                key={symbol}
                type="button"
                onClick={() => setSuitFilter(idx)}
                className={`rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold transition-colors ${
                  suitFilter === idx
                    ? `${SUIT_COLORS[idx].bg} ${SUIT_COLORS[idx].text} shadow-xs border ${SUIT_COLORS[idx].border}`
                    : 'text-muted hover:text-fg'
                }`}
                title={`Filter by ${SUIT_NAMES[idx]}`}
              >
                {symbol}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-1">
            <span className="text-[10px] uppercase text-muted font-semibold">
              Sort:
            </span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'default' | 'ev' | 'weight')}
              className="rounded border border-border bg-surface px-1 py-0.5 text-[10px] text-fg outline-none"
            >
              <option value="default">Default</option>
              <option value="ev">Highest EV</option>
              <option value="weight">Weight</option>
            </select>
          </div>
        </div>
      )}

      {/* Scrollable Combos List */}
      <div
        className="flex-1 divide-y divide-border/60 overflow-y-auto p-2"
        role="list"
        aria-label="Combinations"
      >
        {filteredCombos.map((combo) => (
          <ComboRow
            key={combo.key}
            combo={combo}
            colors={colors}
          />
        ))}

        {filteredCombos.length === 0 && (
          <div className="p-4 text-center text-xs text-muted">
            No combinations match the selected filter.
          </div>
        )}
      </div>
    </aside>
  );
}

function ComboRow({
  combo,
  colors,
}: {
  combo: FormattedCombo;
  colors: Record<string, string>;
}) {
  const isBlocked = combo.isBlocked;
  const isZeroWeight = !isBlocked && combo.weight <= 0.001;

  return (
    <div
      role="listitem"
      data-solver-combo-row
      data-combo-key={combo.key}
      className={`flex flex-col gap-1.5 py-2 px-1 rounded-md transition-colors ${
        isBlocked
          ? 'bg-surface-2/20 opacity-60'
          : isZeroWeight
            ? 'opacity-50'
            : 'hover:bg-surface-2/40'
      }`}
    >
      {/* Top row: Card badges + Stats tags */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <div className="flex items-center gap-1">
            <CardBadge
              rank={combo.card0Rank}
              suit={combo.card0Suit}
              isBlocked={isBlocked}
            />
            <CardBadge
              rank={combo.card1Rank}
              suit={combo.card1Suit}
              isBlocked={isBlocked}
            />
          </div>

          {isBlocked && (
            <span className="inline-flex items-center gap-1 rounded bg-raise/10 px-1.5 py-0.5 text-[10px] font-medium text-raise">
              <ShieldAlert className="h-3 w-3" aria-hidden="true" />
              Blocked by board
            </span>
          )}

          {isZeroWeight && !isBlocked && (
            <span className="rounded bg-surface-2 px-1.5 py-0.5 text-[10px] text-muted">
              Not in range
            </span>
          )}
        </div>

        {/* Combo Metrics */}
        {!isBlocked && (
          <div className="flex items-center gap-2 text-xs">
            {combo.equity > 0 && (
              <span className="font-mono text-[11px] text-muted" title="Equity vs opponent range">
                Eq <strong className="font-semibold text-fg">{formatPercent(combo.equity)}</strong>
              </span>
            )}
            <span
              className={`font-mono text-[11px] font-semibold ${
                combo.ev > 0
                  ? 'text-accent'
                  : combo.ev < 0
                    ? 'text-raise'
                    : 'text-muted'
              }`}
              title="Expected value in big blinds"
            >
              {combo.ev >= 0 ? `+${combo.ev.toFixed(2)}` : combo.ev.toFixed(2)}bb
            </span>
            <span
              className="font-mono text-[10px] text-muted"
              title="Combo weight in range"
            >
              wt: {(combo.weight * 100).toFixed(0)}%
            </span>
          </div>
        )}
      </div>

      {/* Action frequency breakdown bar for this combo */}
      {!isBlocked && combo.actions.length > 0 && (
        <div className="flex flex-col gap-1">
          <div className="flex h-2 w-full overflow-hidden rounded-full bg-surface-2">
            {combo.actions.map((action) => {
              if (action.freq < 0.005) return null;
              return (
                <div
                  key={action.action}
                  style={{
                    width: `${action.freq * 100}%`,
                    backgroundColor: colors[action.action] || 'rgb(var(--muted))',
                  }}
                  title={`${action.action}: ${formatPercent(action.freq)}`}
                />
              );
            })}
          </div>

          <div className="flex flex-wrap items-center gap-x-2 text-[10px] text-muted">
            {combo.actions
              .filter((a) => a.freq >= 0.005)
              .map((action) => (
                <span key={action.action} className="inline-flex items-center gap-1">
                  <span
                    className="h-1.5 w-1.5 rounded-full"
                    style={{
                      backgroundColor:
                        colors[action.action] || 'rgb(var(--muted))',
                    }}
                    aria-hidden="true"
                  />
                  <span>{action.action}</span>
                  <span className="font-mono font-semibold text-fg">
                    {formatPercent(action.freq)}
                  </span>
                </span>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
