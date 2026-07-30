'use client';

import { Check, CircleDot, ScanLine } from 'lucide-react';
import { useEffect, useState } from 'react';
import { HandMatrix, type StrategySegment } from '@/components/hand-matrix/HandMatrix';
import {
  openingSizeLabel,
  type PreflopScenario,
} from '@/data/preflop/catalog';
import type { PreflopChart } from '@/data/preflop/ranges';
import {
  positionFullForSeats,
  positionLabelForSeats,
  TABLE_FORMATS,
  type Position,
  type TableFormat,
} from '@/lib/positions';

type SummaryItem = { name: string; color: string; pct: number };

export interface PreflopWorkspaceProps {
  format: TableFormat;
  hero: Position;
  villain: Position;
  scenarios: PreflopScenario[];
  scenario?: PreflopScenario;
  available: readonly PreflopChart[];
  active?: PreflopChart;
  strategy: Record<string, StrategySegment[]>;
  summary: SummaryItem[];
  onFormat: (format: TableFormat) => void;
  onMatchup: (hero: Position, villain: Position) => void;
  onScenario: (id: string) => void;
}

function Eyebrow({ children }: { children: React.ReactNode }) {
  return (
    <p className="font-mono text-[11px] font-semibold uppercase tracking-[0.16em] text-accent">
      {children}
    </p>
  );
}

function FormatTabs({
  format,
  onFormat,
}: Pick<PreflopWorkspaceProps, 'format' | 'onFormat'>) {
  return (
    <div
      className="grid grid-cols-3 rounded-md border border-border bg-surface p-1"
      aria-label="Table format"
    >
      {TABLE_FORMATS.map((option) => (
        <button
          key={option.seats}
          type="button"
          aria-pressed={format.seats === option.seats}
          onClick={() => onFormat(option)}
          className={
            'min-h-11 rounded px-2 text-sm font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ' +
            (format.seats === option.seats
              ? 'bg-fg text-bg'
              : 'text-muted hover:bg-surface-2 hover:text-fg')
          }
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

function ScenarioControl({
  scenarios,
  scenario,
  onScenario,
}: Pick<PreflopWorkspaceProps, 'scenarios' | 'scenario' | 'onScenario'>) {
  if (!scenario) return null;
  return (
    <div className="min-w-0">
      <div className="flex items-center gap-2">
        <CircleDot className="h-3.5 w-3.5 shrink-0 text-accent" aria-hidden="true" />
        <p className="truncate text-sm font-semibold text-fg">
          {scenario.effectiveStackBb}bb · {openingSizeLabel(scenario.openingSize)}
        </p>
      </div>
      {scenarios.length > 1 && (
        <div
          className="mt-2 flex min-w-0 gap-1 overflow-x-auto pb-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
          aria-label="Preflop scenario"
        >
          {scenarios.map((candidate) => (
            <button
              key={candidate.id}
              type="button"
              aria-pressed={candidate.id === scenario.id}
              onClick={() => onScenario(candidate.id)}
              className={
                'min-h-11 shrink-0 rounded-md border px-3 py-2 text-xs font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ' +
                (candidate.id === scenario.id
                  ? 'border-accent bg-accent/10 text-fg'
                  : 'border-border text-muted hover:bg-surface-2 hover:text-fg')
              }
            >
              {candidate.effectiveStackBb}bb
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function ActionLegend({ summary }: Pick<PreflopWorkspaceProps, 'summary'>) {
  return (
    <div
      className="flex flex-wrap gap-x-4 gap-y-2"
      aria-label="Overall action frequency"
    >
      {summary.map((item) => (
        <span key={item.name} className="flex items-center gap-2 text-xs">
          <span
            className="h-2.5 w-2.5 shrink-0 rounded-[2px]"
            style={{ background: item.color }}
            aria-hidden="true"
          />
          <span className="text-muted">
            {item.name}{' '}
            <strong className="font-mono font-semibold tabular-nums text-fg">
              {item.pct.toFixed(0)}%
            </strong>
          </span>
        </span>
      ))}
    </div>
  );
}

function handMix(
  strategy: Record<string, StrategySegment[]>,
  label: string
): StrategySegment[] {
  const segments = (strategy[label] ?? []).filter(
    (segment) => segment.fraction > 0
  );
  const accountedFor = segments.reduce(
    (total, segment) => total + segment.fraction,
    0
  );

  if (accountedFor >= 1 - 1e-6) return segments;

  return [
    ...segments,
    {
      color: 'rgb(var(--fold))',
      fraction: Math.max(0, 1 - accountedFor),
      label: 'Fold',
    },
  ];
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

function handMixDescription(
  strategy: Record<string, StrategySegment[]>,
  label: string
): string {
  return handMix(strategy, label)
    .map(
      (segment, index) =>
        `${segment.label ?? `Action ${index + 1}`} ${formatProbability(segment.fraction)}`
    )
    .join(', ');
}

function HandMixReadout({
  label,
  strategy,
}: {
  label: string | null;
  strategy: Record<string, StrategySegment[]>;
}) {
  const segments = label ? handMix(strategy, label) : [];

  return (
    <div
      className="flex min-h-5 max-w-full flex-wrap items-center gap-x-3 gap-y-1 text-xs sm:justify-end"
      role="status"
      aria-live="polite"
      aria-atomic="true"
    >
      {label ? (
        <>
          <span className="font-mono font-bold text-accent">{label}</span>
          {segments.map((segment, index) => (
            <span
              key={`${segment.label ?? 'action'}-${index}`}
              className="flex items-center gap-1.5 whitespace-nowrap text-muted"
            >
              <span
                className="h-2 w-2 shrink-0 rounded-[2px]"
                style={{ background: segment.color }}
                aria-hidden="true"
              />
              <span>
                {segment.label ?? `Action ${index + 1}`}{' '}
                <strong className="font-mono font-semibold tabular-nums text-fg">
                  {formatProbability(segment.fraction)}
                </strong>
              </span>
            </span>
          ))}
        </>
      ) : (
        <span className="text-muted">Select a hand to inspect its mix</span>
      )}
    </div>
  );
}

function MatrixPanel({
  active,
  format,
  strategy,
  summary,
}: Pick<
  PreflopWorkspaceProps,
  'active' | 'format' | 'strategy' | 'summary'
>) {
  const [selectedHand, setSelectedHand] = useState<string | null>(null);

  useEffect(() => {
    setSelectedHand(null);
  }, [active?.id]);

  return (
    <section className="order-2 flex min-h-0 flex-col border border-border bg-surface p-3 lg:order-1">
      <div className="flex flex-col gap-3 border-b border-border pb-3 sm:flex-row sm:items-end sm:justify-between">
        <div className="min-w-0">
          <Eyebrow>Preflop library · matrix</Eyebrow>
          <h1 className="mt-1 truncate text-lg font-semibold text-fg">
            {active?.title ?? 'No range for this matchup'}
          </h1>
          {active && (
            <p className="mt-0.5 truncate text-xs text-muted">
              {positionFullForSeats(active.hero, format.seats)}
              {active.vs
                ? ` vs ${positionFullForSeats(active.vs, format.seats)}`
                : ''}
            </p>
          )}
        </div>
        {active && (
          <div className="flex min-w-0 shrink-0 flex-col gap-2 sm:items-end">
            <ActionLegend summary={summary} />
            <HandMixReadout label={selectedHand} strategy={strategy} />
          </div>
        )}
      </div>
      <div className="flex min-h-0 flex-1 items-center justify-center py-3">
        {active ? (
          <div
            key={active.id}
            className="preflop-range-enter mx-auto w-full lg:w-[min(100%,calc(100svh-18.5rem),38rem)]"
            data-preflop-matrix
          >
            <HandMatrix
              mode="display"
              strategy={strategy}
              selectedLabel={selectedHand ?? undefined}
              cellDescription={(label) =>
                handMixDescription(strategy, label)
              }
              onCellClick={(label) =>
                setSelectedHand((current) =>
                  current === label ? null : label
                )
              }
            />
          </div>
        ) : (
          <div className="grid min-h-80 place-items-center text-center text-sm text-muted">
            <div>
              <p className="font-semibold text-fg">No chart for this matchup</p>
              <p className="mt-1">Choose another hero or opposing seat.</p>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

function rangeLabel(
  active: PreflopChart | undefined,
  format: TableFormat
): string {
  if (!active) return 'No bundled range';
  if (active.category === 'RFI') return 'Raise first in';
  return active.vs
    ? `Respond vs ${positionLabelForSeats(active.vs, format.seats)}`
    : 'Response range';
}

function MatchupReadout({
  format,
  hero,
  villain,
  active,
}: Pick<PreflopWorkspaceProps, 'format' | 'hero' | 'villain' | 'active'>) {
  return (
    <div className="grid grid-cols-[1fr_1.25fr_1fr] items-center gap-2 border-y border-border py-3">
      <div>
        <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-muted">
          Hero
        </span>
        <p className="mt-1 font-semibold text-call">
          {positionLabelForSeats(hero, format.seats)}
        </p>
      </div>
      <div className="border-x border-border px-2 text-center">
        <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-muted">
          Active range
        </span>
        <p className="mt-1 truncate text-xs font-semibold text-fg">
          {rangeLabel(active, format)}
        </p>
      </div>
      <div className="text-right">
        <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-muted">
          Opponent
        </span>
        <p className="mt-1 font-semibold text-raise">
          {positionLabelForSeats(villain, format.seats)}
        </p>
      </div>
    </div>
  );
}

function hasRangeForMatchup(
  charts: readonly PreflopChart[],
  hero: Position,
  villain: Position
): boolean {
  return charts.some(
    (chart) =>
      chart.hero === hero &&
      (chart.category === 'RFI' || chart.vs === villain)
  );
}

function MatchupGrid({
  format,
  hero,
  villain,
  available,
  onMatchup,
}: Pick<
  PreflopWorkspaceProps,
  'format' | 'hero' | 'villain' | 'available' | 'onMatchup'
>) {
  const positions = format.positions;
  const gridWidth = 44 * (positions.length + 1);

  return (
    <div className="min-h-0 overflow-auto border-y border-border py-3">
      <div
        className="grid gap-1"
        style={{
          width: `${gridWidth}px`,
          gridTemplateColumns: `repeat(${positions.length + 1}, 44px)`,
        }}
        aria-label="Hero and opponent matchup grid"
      >
        <div className="grid h-11 place-items-center">
          <ScanLine className="h-4 w-4 text-accent" aria-hidden="true" />
        </div>
        {positions.map((position) => (
          <div
            key={`column-${position}`}
            className="grid h-11 place-items-center font-mono text-[10px] font-semibold text-raise"
            aria-hidden="true"
          >
            {positionLabelForSeats(position, format.seats)}
          </div>
        ))}

        {positions.map((heroPosition) => (
          <div key={`row-${heroPosition}`} className="contents">
            <div
              className="grid h-11 place-items-center font-mono text-[10px] font-semibold text-call"
              aria-hidden="true"
            >
              {positionLabelForSeats(heroPosition, format.seats)}
            </div>
            {positions.map((villainPosition) => {
              const disabled =
                heroPosition === villainPosition ||
                !hasRangeForMatchup(available, heroPosition, villainPosition);
              const selected =
                heroPosition === hero && villainPosition === villain;
              return (
                <button
                  key={`${heroPosition}-${villainPosition}`}
                  type="button"
                  disabled={disabled}
                  onClick={() => onMatchup(heroPosition, villainPosition)}
                  aria-label={`${positionFullForSeats(heroPosition, format.seats)} hero vs ${positionFullForSeats(villainPosition, format.seats)} opponent${disabled ? ', no bundled range' : ''}`}
                  aria-pressed={selected}
                  className={
                    'grid h-11 w-11 place-items-center rounded-md border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:cursor-not-allowed disabled:border-transparent disabled:bg-surface-2/40 ' +
                    (selected
                      ? 'border-accent bg-accent text-accent-fg'
                      : disabled
                        ? ''
                        : 'border-border bg-surface text-muted hover:bg-surface-2 hover:text-fg')
                  }
                >
                  {selected ? (
                    <Check className="h-4 w-4" aria-hidden="true" />
                  ) : disabled ? (
                    <span className="h-px w-3 bg-border" aria-hidden="true" />
                  ) : (
                    <span className="h-1.5 w-1.5 rounded-full bg-current" aria-hidden="true" />
                  )}
                </button>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

export function PreflopWorkspace(props: PreflopWorkspaceProps) {
  return (
    <div className="preflop-workspace grid gap-3 lg:h-[calc(100svh-7rem)] lg:grid-cols-[minmax(0,1.35fr)_minmax(380px,0.65fr)] lg:overflow-hidden">
      <MatrixPanel {...props} />
      <aside className="order-1 min-h-0 overflow-y-auto lg:order-2">
        <div className="flex min-h-full flex-col gap-3 lg:h-full">
          <header className="border-b border-border pb-3">
            <Eyebrow>Matchup selector · Hero × Opponent</Eyebrow>
            <h2 className="mt-1 text-2xl font-semibold tracking-tight text-fg">
              Choose both seats
            </h2>
            <p className="mt-1 text-sm leading-5 text-muted">
              Rows are Hero and columns are the opponent. The closest bundled
              response range is selected automatically.
            </p>
          </header>
          <FormatTabs {...props} />
          <ScenarioControl {...props} />
          <MatchupGrid {...props} />
          <MatchupReadout {...props} />
        </div>
      </aside>
    </div>
  );
}
