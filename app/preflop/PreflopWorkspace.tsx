'use client';

import { useEffect, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import {
  HandMatrix,
  type StrategySegment,
} from '@/components/hand-matrix/HandMatrix';
import { PageHeading } from '@/components/design/DesignShell';
import { Select } from '@/components/ui/Select';
import { openingSizeLabel, type PreflopScenario } from '@/data/preflop/catalog';
import type { PreflopChart } from '@/data/preflop/ranges';
import { cardRank, cardSuit, comboLabelToCombos, RANKS } from '@/lib/cards';
import { SUIT_SYMBOLS } from '@/lib/combo-breakdown';
import {
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
  onChart: (id: string) => void;
}

function handMix(strategy: Record<string, StrategySegment[]>, label: string) {
  const segments = (strategy[label] ?? []).filter(
    (segment) => segment.fraction > 0
  );
  const remainder =
    1 - segments.reduce((sum, segment) => sum + segment.fraction, 0);
  return remainder > 1e-6
    ? [
        ...segments,
        { color: 'rgb(var(--fold))', fraction: remainder, label: 'Fold' },
      ]
    : segments;
}

function probability(fraction: number) {
  const percentage = fraction * 100;
  return percentage > 0 && percentage < 0.1
    ? '<0.1%'
    : `${Math.round(percentage * 10) / 10}%`;
}

function HandDetail({
  label,
  strategy,
}: {
  label: string;
  strategy: Record<string, StrategySegment[]>;
}) {
  const mix = handMix(strategy, label);
  const combos = comboLabelToCombos(label);
  const suitClasses = [
    'text-emerald-950',
    'text-blue-950',
    'text-red-950',
    'text-zinc-950',
  ];
  return (
    <section
      className="preflop-hand-detail"
      aria-label={`Combinations for ${label}`}
    >
      <div className="study-section-heading">
        <h2>{label}</h2>
        <span>{combos.length} combos</span>
      </div>
      <div className="preflop-hand-mix" role="status" aria-live="polite">
        {mix.map((segment, index) => (
          <span key={index}>
            <i style={{ background: segment.color }} aria-hidden="true" />
            {segment.label} <strong>{probability(segment.fraction)}</strong>
          </span>
        ))}
      </div>
      <div className="preflop-combos">
        {combos.map((combo) => (
          <div
            key={combo.join('-')}
            className="preflop-combo"
            aria-label={`${combo.map((card) => `${RANKS[cardRank(card)]}${SUIT_SYMBOLS[cardSuit(card)]}`).join(' ')}: ${mix.map((segment) => `${segment.label} ${probability(segment.fraction)}`).join(', ')}`}
          >
            <span className="absolute inset-0 flex" aria-hidden="true">
              {mix.map((segment, index) => (
                <span
                  key={index}
                  style={{
                    width: `${segment.fraction * 100}%`,
                    background: segment.color,
                  }}
                />
              ))}
            </span>
            {combo.map((card) => (
              <span
                key={card}
                className={`relative z-10 inline-flex gap-0.5 font-semibold ${suitClasses[cardSuit(card)]}`}
              >
                {RANKS[cardRank(card)]}
                <span>{SUIT_SYMBOLS[cardSuit(card)]}</span>
              </span>
            ))}
          </div>
        ))}
      </div>
    </section>
  );
}

export function PreflopWorkspace(props: PreflopWorkspaceProps) {
  const { format, hero, villain, active, scenario, strategy, summary } = props;
  const [pinned, setPinned] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [hovered, setHovered] = useState('AKs');
  useEffect(() => {
    setPinned(null);
    setHovered('AKs');
  }, [active?.id]);
  const charts = props.available.filter(
    (chart) =>
      chart.hero === hero && (chart.category === 'RFI' || chart.vs === villain)
  );
  const positions = format.positions.map((position) => ({
    value: position,
    label: positionLabelForSeats(position, format.seats),
  }));

  return (
    <div
      className="preflop-study"
      onKeyDown={(event) => {
        if (event.key === 'Escape') setPinned(null);
      }}
    >
      <PageHeading
        title="Range library"
        meta={`${format.label} · ${scenario?.effectiveStackBb ?? 100} bb`}
      />
      <div className="preflop-layout">
        <section className="preflop-config" aria-label="Range settings">
          <button
            type="button"
            className="preflop-config-toggle"
            aria-label="Configure range"
            aria-expanded={settingsOpen}
            aria-controls="preflop-controls"
            onClick={() => setSettingsOpen(!settingsOpen)}
          >
            <span>
              {positionLabelForSeats(hero, format.seats)} vs{' '}
              {positionLabelForSeats(villain, format.seats)}
            </span>
            <span className="inline-flex items-center gap-2">
              Configure <ChevronDown className="h-4 w-4" aria-hidden="true" />
            </span>
          </button>
          <div
            id="preflop-controls"
            className="preflop-controls"
            data-open={settingsOpen}
          >
            <Select
              label="Table"
              value={String(format.seats)}
              options={TABLE_FORMATS.map((option) => ({
                value: String(option.seats),
                label: option.label,
              }))}
              onChange={(value) => {
                const next = TABLE_FORMATS.find(
                  (option) => String(option.seats) === value
                );
                if (next) props.onFormat(next);
              }}
            />
            <Select
              label="Stack"
              value={scenario?.id ?? ''}
              options={props.scenarios.map((option) => ({
                value: option.id,
                label: `${option.effectiveStackBb} bb`,
                description: openingSizeLabel(option.openingSize),
              }))}
              onChange={props.onScenario}
            />
            <Select
              label="Hero"
              value={hero}
              options={positions}
              onChange={(value) =>
                props.onMatchup(
                  value as Position,
                  value === villain
                    ? format.positions.find((position) => position !== value)!
                    : villain
                )
              }
            />
            <Select
              label="Opponent"
              value={villain}
              options={positions.map((option) => ({
                ...option,
                disabled: option.value === hero,
              }))}
              onChange={(value) => props.onMatchup(hero, value as Position)}
            />
            <Select
              label="Situation"
              value={active?.id ?? ''}
              options={charts.map((chart) => ({
                value: chart.id,
                label:
                  chart.category === 'RFI'
                    ? 'Open raise'
                    : chart.category === 'vs-RFI'
                      ? 'Facing a raise'
                      : chart.title,
              }))}
              onChange={props.onChart}
            />
          </div>
        </section>
        <section className="preflop-matrix-panel" aria-label="Range matrix">
          <div className="study-section-heading">
            <h2>{active?.title ?? 'No range available'}</h2>
            <span>
              {scenario ? openingSizeLabel(scenario.openingSize) : ''}
            </span>
          </div>
          <div className="preflop-legend" aria-label="Overall action frequency">
            {summary.map((item) => (
              <span key={item.name}>
                <i style={{ background: item.color }} aria-hidden="true" />
                {item.name} <strong>{Math.round(item.pct)}%</strong>
              </span>
            ))}
          </div>
          {active ? (
            <div className="preflop-matrix" data-preflop-matrix>
              <HandMatrix
                squareCells
                mode="display"
                strategy={strategy}
                selectedLabel={pinned ?? undefined}
                onCellPreview={setHovered}
                cellDescription={(label) =>
                  handMix(strategy, label)
                    .map(
                      (segment) =>
                        `${segment.label} ${probability(segment.fraction)}`
                    )
                    .join(', ')
                }
                onCellClick={(label) =>
                  setPinned((current) => (current === label ? null : label))
                }
              />
            </div>
          ) : (
            <div className="study-empty">
              <h2>No chart for these seats</h2>
              <p>Choose another matchup.</p>
            </div>
          )}
        </section>
        <aside className="preflop-details">
          {active && (
            <HandDetail label={pinned ?? hovered} strategy={strategy} />
          )}
          <section
            className="preflop-seat-ranges"
            aria-label="Opening ranges by seat"
          >
            <div className="study-section-heading">
              <h2>Opening ranges</h2>
              <span>{format.label}</span>
            </div>
            <div>
              {format.positions.map((position) => (
                <button
                  type="button"
                  key={position}
                  aria-pressed={hero === position && active?.category === 'RFI'}
                  onClick={() => {
                    const chart = props.available.find(
                      (entry) =>
                        entry.hero === position && entry.category === 'RFI'
                    );
                    if (chart) {
                      props.onMatchup(
                        position,
                        position === villain
                          ? format.positions.find((seat) => seat !== position)!
                          : villain
                      );
                      props.onChart(chart.id);
                    }
                  }}
                  disabled={
                    !props.available.some(
                      (entry) =>
                        entry.hero === position && entry.category === 'RFI'
                    )
                  }
                >
                  {positionLabelForSeats(position, format.seats)}
                </button>
              ))}
            </div>
          </section>
          {scenario && (
            <details className="study-disclosure">
              <summary>
                Model · {scenario.provenance.status}
                <ChevronDown className="h-4 w-4" aria-hidden="true" />
              </summary>
              <div>
                <p>{scenario.provenance.model}</p>
                <ul>
                  {scenario.provenance.assumptions.map((assumption) => (
                    <li key={assumption}>{assumption}</li>
                  ))}
                </ul>
              </div>
            </details>
          )}
        </aside>
      </div>
    </div>
  );
}
