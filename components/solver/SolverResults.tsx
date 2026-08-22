'use client';

import { useEffect, useMemo, useState } from 'react';
import { HandMatrix, StrategySegment } from '@/components/hand-matrix/HandMatrix';
import type {
  ActionStrategy,
  ClassRow,
  NodeStrategy,
  SolverResult,
} from '@/lib/solver/client';

// Deterministic colors per action label, anchored to poker conventions.
const BET_RAMP = ['#f59e0b', '#f97316', '#dc2626'];
const RAISE_RAMP = ['#a855f7', '#7c3aed'];

function colorForActions(labels: string[]): Record<string, string> {
  const map: Record<string, string> = {};
  let betI = 0;
  let raiseI = 0;
  for (const l of labels) {
    if (l.startsWith('Fold')) map[l] = 'rgb(var(--fold))';
    else if (l.startsWith('Check')) map[l] = 'rgb(var(--check))';
    else if (l.startsWith('Call')) map[l] = 'rgb(var(--call))';
    else if (l.startsWith('Raise'))
      map[l] = RAISE_RAMP[Math.min(raiseI++, RAISE_RAMP.length - 1)];
    else if (l.startsWith('Bet'))
      map[l] = BET_RAMP[Math.min(betI++, BET_RAMP.length - 1)];
    else map[l] = 'rgb(var(--muted))';
  }
  return map;
}

function toStrategy(
  node: NodeStrategy,
  colors: Record<string, string>
): Record<string, StrategySegment[]> {
  const out: Record<string, StrategySegment[]> = {};
  for (const row of node.rows) {
    const segs: StrategySegment[] = [];
    for (const a of row.actions) {
      if (a.freq > 0.005)
        segs.push({ color: colors[a.action], fraction: a.freq, label: a.action });
    }
    if (segs.length) out[row.class] = segs;
  }
  return out;
}

function formatProbability(fraction: number): string {
  if (fraction <= 0) return '0%';
  if (fraction >= 1) return '100%';

  const percentage = fraction * 100;
  if (percentage < 0.05) return '<0.1%';
  if (percentage > 99.95) return '>99.9%';

  const rounded = Math.round(percentage * 10) / 10;
  return Number.isInteger(rounded)
    ? `${rounded.toFixed(0)}%`
    : `${rounded.toFixed(1)}%`;
}

function handDescription(row: ClassRow): string {
  const actions = row.actions
    .map((action) => `${action.action} ${formatProbability(action.freq)}`)
    .join(', ');
  const ev = row.actions[0]?.ev;
  return `${actions}${ev === undefined ? '' : `, hand-class EV ${ev.toFixed(2)}bb`}`;
}

function HandMixReadout({
  label,
  actions,
  colors,
}: {
  label: string | null;
  actions: ActionStrategy[];
  colors: Record<string, string>;
}) {
  const ev = actions[0]?.ev;

  return (
    <div
      className="mb-3 flex min-h-5 max-w-full flex-wrap items-center gap-x-3 gap-y-1 text-xs"
      role="status"
      aria-live="polite"
      aria-atomic="true"
      data-solver-hand-mix
    >
      {label ? (
        <>
          <span className="font-mono font-bold text-accent">{label}</span>
          {actions.map((action) => (
            <span
              key={action.action}
              className="flex items-center gap-1.5 whitespace-nowrap text-muted"
            >
              <span
                className="h-2 w-2 shrink-0 rounded-[2px]"
                style={{ background: colors[action.action] }}
                aria-hidden="true"
              />
              <span>
                {action.action}{' '}
                <strong className="font-mono font-semibold tabular-nums text-fg">
                  {formatProbability(action.freq)}
                </strong>
              </span>
            </span>
          ))}
          {ev !== undefined && (
            <span className="whitespace-nowrap text-muted">
              EV{' '}
              <strong className="font-mono font-semibold tabular-nums text-fg">
                {ev.toFixed(2)}bb
              </strong>
            </span>
          )}
        </>
      ) : (
        <span className="text-muted">Select a hand to inspect its mix</span>
      )}
    </div>
  );
}

export function StrategyView({ node }: { node: NodeStrategy }) {
  const [selectedHand, setSelectedHand] = useState<string | null>(null);
  const colors = useMemo(() => colorForActions(node.actions), [node.actions]);
  const strategy = useMemo(() => toStrategy(node, colors), [node, colors]);
  const rowsByClass = useMemo(
    () => new Map(node.rows.map((row) => [row.class, row])),
    [node.rows]
  );
  const selectedRow = selectedHand ? rowsByClass.get(selectedHand) : undefined;
  const annotation = useMemo(() => {
    const evByClass: Record<string, number> = {};
    for (const r of node.rows) evByClass[r.class] = r.actions[0]?.ev ?? 0;
    return (label: string) =>
      evByClass[label] !== undefined ? evByClass[label].toFixed(1) : undefined;
  }, [node.rows]);

  useEffect(() => {
    setSelectedHand(null);
  }, [node]);

  if (!node.rows.length) return null;

  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="text-sm font-semibold">{node.title}</div>
        <div className="flex flex-wrap justify-end gap-2 text-xs">
          {node.actions.map((a) => (
            <span key={a} className="flex items-center gap-1.5">
              <span
                className="h-3 w-3 rounded-sm"
                style={{ background: colors[a] }}
              />
              {a}
            </span>
          ))}
        </div>
      </div>
      <HandMixReadout
        label={selectedRow?.class ?? null}
        actions={selectedRow?.actions ?? []}
        colors={colors}
      />
      <div className="w-full">
        <HandMatrix
          mode="display"
          strategy={strategy}
          annotation={annotation}
          selectedLabel={selectedRow?.class}
          cellDescription={(label) => {
            const row = rowsByClass.get(label);
            return row ? handDescription(row) : undefined;
          }}
          onCellClick={(label) =>
            setSelectedHand((current) => (current === label ? null : label))
          }
        />
      </div>
      <p className="mt-2 text-[11px] text-muted">
        Select a hand for exact action frequencies; the small number is its EV
        (bb).
      </p>
    </div>
  );
}

/** The compact summary bar (exploitability + EVs). Shown at the top so a
 * result is always immediately visible, and surfaces solver errors. */
export function SolverStats({ result }: { result: SolverResult }) {
  if (result.error) {
    return (
      <div
        role="alert"
        className="rounded-lg border border-raise/40 bg-raise/10 p-4 text-sm text-raise"
      >
        Solver error: {result.error}
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Exploitability" value={`${result.exploitability_pct}%`} hint="of pot" />
        <Stat label="OOP EV" value={`${result.oop_ev} bb`} />
        <Stat label="IP EV" value={`${result.ip_ev} bb`} />
        <Stat label="Iterations" value={`${result.iterations}`} />
      </div>
      {result.truncated && (
        <div className="rounded-md border border-border bg-surface-2 p-2 text-xs text-muted">
          Ranges were capped to keep the solve fast — results use the
          highest-weight combos.
        </div>
      )}
    </div>
  );
}

/** The strategy grids for both players. */
export function SolverStrategies({ result }: { result: SolverResult }) {
  if (result.error) return null;
  return (
    <div className="flex flex-col gap-4">
      <StrategyView node={result.oop} />
      <StrategyView node={result.ip} />
    </div>
  );
}

/** Combined view (stats + strategies), kept for convenience. */
export function SolverResults({ result }: { result: SolverResult }) {
  return (
    <div className="flex flex-col gap-4">
      <SolverStats result={result} />
      <SolverStrategies result={result} />
    </div>
  );
}

function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="text-xs text-muted">{label}</div>
      <div className="text-lg font-semibold tabular-nums">
        {value}
        {hint && <span className="ml-1 text-xs font-normal text-muted">{hint}</span>}
      </div>
    </div>
  );
}
