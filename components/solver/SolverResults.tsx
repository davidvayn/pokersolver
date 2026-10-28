'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import {
  HandMatrix,
  StrategySegment,
} from '@/components/hand-matrix/HandMatrix';
import { ComboInspector } from '@/components/solver/ComboInspector';
import { handClassLabel, type Card } from '@/lib/cards';
import { compactStrategy } from '@/lib/solver/strategy-display';
import type {
  ActionStrategy,
  ClassRow,
  NodeStrategy,
  SolverResult,
} from '@/lib/solver/client';

// Deterministic colors per action label, anchored to poker conventions.
const BET_RAMP = ['#f59e0b', '#f97316', '#dc2626'];
const RAISE_RAMP = ['#a855f7', '#7c3aed'];

export function colorForActions(labels: string[]): Record<string, string> {
  const map: Record<string, string> = {};
  let betI = 0;
  let raiseI = 0;
  for (const l of labels) {
    if (l.startsWith('Fold')) map[l] = 'rgb(var(--fold))';
    else if (l.startsWith('Check')) map[l] = 'rgb(var(--check))';
    else if (l.startsWith('Call')) map[l] = 'rgb(var(--call))';
    else if (l.startsWith('All-in')) map[l] = 'rgb(var(--allin))';
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
        segs.push({
          color: colors[a.action],
          fraction: a.freq,
          label: a.action,
        });
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
  const ev = row.ev === undefined ? row.actions[0]?.ev : row.ev;
  return `${actions}${ev == null ? '' : `, hand-class EV ${ev.toFixed(2)}bb`}`;
}

function HandMixReadout({
  label,
  actions,
  colors,
  policyEv,
}: {
  label: string | null;
  actions: ActionStrategy[];
  colors: Record<string, string>;
  policyEv?: number | null;
}) {
  const ev = policyEv === undefined ? actions[0]?.ev : policyEv;

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
          {ev != null && (
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

export function StrategyView({
  node: sourceNode,
  board,
  framed = true,
  compact = false,
  matrixClassName = '',
  settings,
  pending = false,
  completed = false,
  compactActions = false,
  fitViewport = false,
}: {
  node: NodeStrategy;
  board?: Card[];
  framed?: boolean;
  compact?: boolean;
  matrixClassName?: string;
  settings?: ReactNode;
  pending?: boolean;
  completed?: boolean;
  compactActions?: boolean;
  fitViewport?: boolean;
}) {
  const node = useMemo(
    () => (compactActions ? compactStrategy(sourceNode) : sourceNode),
    [sourceNode, compactActions]
  );
  const matrixFrame = useRef<HTMLDivElement>(null);
  const strategyLayout = useRef<HTMLDivElement>(null);
  const [selection, setSelection] = useState<{
    completed: boolean;
    hand: string | null;
    preview: string | null;
  }>({ completed, hand: null, preview: null });
  const selectedHand =
    selection.completed === completed ? selection.hand : null;
  const previewHand =
    selection.completed === completed ? selection.preview : null;
  const lastInspectedHand = useRef<string | undefined>(undefined);
  const [showMetrics, setShowMetrics] = useState(false);
  const colors = useMemo(() => colorForActions(node.actions), [node.actions]);
  const strategy = useMemo(() => toStrategy(node, colors), [node, colors]);
  const rowsByClass = useMemo(
    () => new Map(node.rows.map((row) => [row.class, row])),
    [node.rows]
  );
  const firstHand = useMemo(() => {
    for (let row = 0; row < 13; row++) {
      for (let column = 0; column < 13; column++) {
        const label = handClassLabel(row, column);
        if (rowsByClass.has(label)) return label;
      }
    }
    return undefined;
  }, [rowsByClass]);
  const activeHand = pending
    ? undefined
    : (selectedHand ?? previewHand ?? (completed ? undefined : firstHand));
  const rememberedHand = lastInspectedHand.current;
  const inspectedHand =
    activeHand ??
    (!pending && completed
      ? rememberedHand && rowsByClass.has(rememberedHand)
        ? rememberedHand
        : firstHand
      : undefined);
  const selectedRow = activeHand ? rowsByClass.get(activeHand) : undefined;
  const hiddenSizes =
    compactActions && sourceNode.actions.length > node.actions.length;
  const metricsActive = showMetrics && (!completed || Boolean(activeHand));
  const annotation = useMemo(() => {
    const evByClass: Record<string, number> = {};
    for (const r of node.rows) {
      const ev = r.ev === undefined ? r.actions[0]?.ev : r.ev;
      if (ev != null) evByClass[r.class] = ev;
    }
    return (label: string) =>
      evByClass[label] !== undefined ? evByClass[label].toFixed(1) : undefined;
  }, [node.rows]);

  const metricsToggle = (
    <button
      type="button"
      disabled={pending || (completed && !activeHand)}
      aria-pressed={metricsActive}
      onClick={() => setShowMetrics((shown) => !shown)}
      className="min-h-9 shrink-0 text-xs text-muted underline-offset-4 hover:text-fg aria-pressed:text-accent aria-pressed:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
    >
      EV / equity
    </button>
  );

  useEffect(() => {
    if (!fitViewport) return;
    const frame = matrixFrame.current;
    const layout = strategyLayout.current;
    if (!frame || !layout) return;
    const footer = frame.nextElementSibling;
    function measure() {
      if (!frame || !layout) return;
      if (window.innerWidth < 768) {
        layout.style.removeProperty('--solver-matrix-size');
        layout.style.removeProperty('--solver-review-height');
        return;
      }
      const footerSpace = footer
        ? footer.getBoundingClientRect().height +
          parseFloat(getComputedStyle(footer).marginTop)
        : 0;
      const room = Math.max(
        240,
        Math.floor(
          window.innerHeight -
            frame.getBoundingClientRect().top -
            window.scrollY -
            footerSpace -
            40
        )
      );
      layout.style.setProperty('--solver-matrix-size', `${room}px`);
      layout.style.setProperty(
        '--solver-review-height',
        `${Math.max(240, window.innerHeight - layout.getBoundingClientRect().top - window.scrollY - 40)}px`
      );
    }
    let animation = 0;
    function fit() {
      cancelAnimationFrame(animation);
      animation = requestAnimationFrame(measure);
    }
    const observer = new ResizeObserver(fit);
    const toolbar = layout
      .closest('[data-solver-workspace]')
      ?.querySelector('.solver-spot-toolbar');
    const heading = layout.querySelector('[data-solver-strategy-heading]');
    if (toolbar) observer.observe(toolbar);
    if (heading) observer.observe(heading);
    if (footer) observer.observe(footer);
    fit();
    window.addEventListener('resize', fit);
    return () => {
      cancelAnimationFrame(animation);
      observer.disconnect();
      window.removeEventListener('resize', fit);
    };
  }, [fitViewport, pending, completed, node.actions]);

  useEffect(() => {
    setSelection((current) => ({
      completed,
      hand:
        current.completed === completed &&
        current.hand &&
        rowsByClass.has(current.hand)
          ? current.hand
          : null,
      preview:
        current.completed === completed &&
        current.preview &&
        rowsByClass.has(current.preview)
          ? current.preview
          : null,
    }));
  }, [rowsByClass, completed]);

  useEffect(() => {
    if (activeHand) lastInspectedHand.current = activeHand;
  }, [activeHand]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setSelection((current) => ({ ...current, hand: null, preview: null }));
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <div
      aria-busy={pending}
      data-solver-line-completed={completed}
      className={
        framed
          ? `rounded-lg border border-border bg-surface ${compact ? 'p-3' : 'p-4'}`
          : 'min-h-0'
      }
    >
      <div ref={strategyLayout} className="solver-strategy-layout">
        {settings && (
          <div className="solver-strategy-settings min-w-0">{settings}</div>
        )}
        <div className={`solver-strategy-main min-w-0 ${matrixClassName}`}>
          <div
            data-solver-strategy-heading
            className={`${compact ? 'mb-2' : 'mb-3'} flex items-center justify-between gap-3`}
          >
            <div className="shrink-0 text-xs text-muted">
              {completed ? `Last decision · ${node.title}` : node.title}
            </div>
            <div className="flex flex-wrap items-center justify-end gap-x-3 gap-y-1 text-xs">
              {!compactActions && metricsToggle}
              {!pending &&
                node.actions.map((a) => (
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
          <div ref={matrixFrame} data-solver-matrix-frame>
            <HandMatrix
              mode="display"
              strategy={pending ? {} : strategy}
              annotation={!pending && metricsActive ? annotation : undefined}
              selectedLabel={activeHand}
              selectionOutline={!compactActions}
              squareCells
              cellDescription={
                pending
                  ? undefined
                  : (label) => {
                      const row = rowsByClass.get(label);
                      return row ? handDescription(row) : 'Not in solved range';
                    }
              }
              onCellPreview={
                pending
                  ? undefined
                  : (label) =>
                      setSelection((current) => ({
                        completed,
                        hand:
                          current.completed === completed ? current.hand : null,
                        preview: label,
                      }))
              }
              onCellClick={
                pending
                  ? undefined
                  : (label) => {
                      setSelection((current) => ({
                        completed,
                        preview: label,
                        hand:
                          current.completed === completed &&
                          current.hand === label
                            ? null
                            : label,
                      }));
                    }
              }
            />
          </div>
          <div className="mt-3">
            {!pending && !compactActions && (
              <HandMixReadout
                label={selectedRow?.class ?? activeHand ?? null}
                actions={selectedRow?.actions ?? []}
                colors={colors}
                policyEv={selectedRow?.ev}
              />
            )}
            {compactActions && (
              <div className="flex items-center justify-between gap-2">
                {metricsToggle}
                {hiddenSizes && !pending && (
                  <p
                    className="inline-flex items-center gap-1.5 text-xs text-muted"
                    title="Original policy frequencies. Hatched space represents the smaller trained sizes hidden from this four-action view."
                  >
                    <span
                      className="solver-hidden-size-swatch h-3 w-3 shrink-0"
                      aria-hidden="true"
                    />
                    Other sizes hidden
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
        {inspectedHand ? (
          <ComboInspector
            key={inspectedHand}
            label={inspectedHand}
            row={rowsByClass.get(inspectedHand)}
            board={board}
            colors={colors}
            showMetrics={metricsActive}
            deselected={completed && !activeHand}
            className="solver-strategy-combos"
          />
        ) : completed && !pending ? (
          <aside
            data-solver-combo-inspector
            data-deselected="true"
            aria-label="Combination details"
            className="solver-strategy-combos min-h-64 text-sm text-muted"
          >
            <p className="border-b border-border pb-2 font-semibold">
              Combination details
            </p>
            <p className="py-4 text-xs">
              No supported combo values for this last decision.
            </p>
          </aside>
        ) : null}
      </div>
      {!compact && (
        <p className="mt-2 text-[11px] text-muted">
          Select a hand for action frequencies and individual combos. Enable EV
          / equity to see additional metrics.
        </p>
      )}
    </div>
  );
}

/** Optional diagnostics kept out of the primary solving flow. */
export function SolverNerdStats({
  result,
  compact = false,
}: {
  result: SolverResult;
  compact?: boolean;
}) {
  if (result.error) return null;

  return (
    <section
      aria-labelledby="solver-nerd-stats-heading"
      className={`rounded-lg border border-border bg-surface ${compact ? 'p-3' : 'p-4'}`}
    >
      <div>
        <h2 id="solver-nerd-stats-heading" className="text-sm font-semibold">
          Stats for nerds
        </h2>
        {!compact && (
          <p className="mt-1 text-xs text-muted">
            Diagnostics for the current abstract solve.
          </p>
        )}
      </div>
      <dl
        className={`${compact ? 'mt-2 gap-x-3 gap-y-2 pt-2' : 'mt-4 gap-x-4 gap-y-4 pt-4'} grid grid-cols-2 border-t border-border sm:grid-cols-3 xl:grid-cols-5`}
      >
        <Stat label="Exploitability" value={`${result.exploitability_pct}%`} />
        <Stat label="OOP EV" value={`${result.oop_ev} bb`} />
        <Stat label="IP EV" value={`${result.ip_ev} bb`} />
        <Stat label="Model" value="CFR+" />
        <Stat label="Iterations" value={`${result.iterations}`} />
      </dl>
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs text-muted">{label}</dt>
      <dd className="mt-1 font-mono text-sm font-semibold tabular-nums text-fg">
        {value}
      </dd>
    </div>
  );
}
