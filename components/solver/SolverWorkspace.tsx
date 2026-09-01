'use client';

import { useId, useState } from 'react';
import { CircleDot, LoaderCircle, Trash2 } from 'lucide-react';
import { AiPanel } from '@/components/ai/AiPanel';
import { GeminiMark } from '@/components/ai/GeminiMark';
import { CardSlots } from '@/components/board/CardPicker';
import { RangeEditor } from '@/components/range/RangeEditor';
import {
  SolverNerdStats,
  StrategyView,
} from '@/components/solver/SolverResults';
import { handClassLabel, type Card } from '@/lib/cards';
import type { SpotContext } from '@/lib/ai/prompt';
import type { SolverResult } from '@/lib/solver/client';

type Player = 'oop' | 'ip';

export interface SolverWorkspaceProps {
  board: Card[];
  used: Set<Card>;
  onBoardChange: (cards: Card[]) => void;
  oop: Record<string, number>;
  ip: Record<string, number>;
  onOopChange: (weights: Record<string, number>) => void;
  onIpChange: (weights: Record<string, number>) => void;
  pot: number;
  stack: number;
  betSizes: string;
  raiseSizes: string;
  onPotChange: (value: number) => void;
  onStackChange: (value: number) => void;
  onBetSizesChange: (value: string) => void;
  onRaiseSizesChange: (value: string) => void;
  result: SolverResult | null;
  running: boolean;
  available: boolean;
  solverError: string | null;
  missing: string | null;
  showSolverStats: boolean;
  getAnalysisSpot: () => SpotContext | null;
  onClear: () => void;
}

interface WorkspaceContext extends SolverWorkspaceProps {
  rangeTab: Player;
  setRangeTab: (player: Player) => void;
  strategyTab: Player;
  setStrategyTab: (player: Player) => void;
}

const OOP_COLOR = 'rgb(var(--check))';
const IP_COLOR = 'rgb(var(--allin))';

function playerColor(player: Player): string {
  return player === 'oop' ? OOP_COLOR : IP_COLOR;
}

function PlayerTabs({
  value,
  onChange,
  suffix = '',
}: {
  value: Player;
  onChange: (player: Player) => void;
  suffix?: string;
}) {
  return (
    <div className="grid grid-cols-2 gap-1 rounded-md bg-surface-2 p-1" role="group">
      {(['oop', 'ip'] as const).map((player) => (
        <button
          key={player}
          type="button"
          onClick={() => onChange(player)}
          aria-pressed={value === player}
          className={`min-h-10 rounded px-2 text-xs font-semibold uppercase transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
            value === player
              ? 'bg-surface text-fg shadow-sm'
              : 'text-muted hover:text-fg'
          }`}
        >
          <span className="inline-flex items-center gap-1.5">
            <span
              className="h-2 w-2 rounded-full"
              style={{ background: playerColor(player) }}
              aria-hidden="true"
            />
            {player.toUpperCase()}
            {suffix}
          </span>
        </button>
      ))}
    </div>
  );
}

function CompactField({
  label,
  value,
  onChange,
  number = false,
}: {
  label: string;
  value: string | number;
  onChange: (value: string) => void;
  number?: boolean;
}) {
  const id = useId();

  return (
    <label
      htmlFor={id}
      className="flex h-11 min-w-0 items-center gap-2 rounded-md border border-border bg-surface-2 px-2"
    >
      <span className="shrink-0 text-[10px] font-semibold uppercase text-muted">
        {label}
      </span>
      <input
        id={id}
        type={number ? 'number' : 'text'}
        inputMode={number ? 'decimal' : 'text'}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="min-w-0 flex-1 bg-transparent text-right font-mono text-xs font-semibold tabular-nums text-fg outline-none"
      />
    </label>
  );
}

function BoardControls({ context }: { context: WorkspaceContext }) {
  return (
    <div className="flex min-w-0 items-center gap-3">
      <CardSlots
        count={5}
        cards={context.board}
        used={context.used}
        onChange={context.onBoardChange}
        size="xl"
      />
      <div className="hidden min-w-0 flex-1 grid-cols-4 gap-2 md:grid">
        <CompactField
          label="Pot"
          value={context.pot}
          number
          onChange={(value) => context.onPotChange(parseFloat(value) || 0)}
        />
        <CompactField
          label="Stack"
          value={context.stack}
          number
          onChange={(value) => context.onStackChange(parseFloat(value) || 0)}
        />
        <CompactField
          label="Bet %"
          value={context.betSizes}
          onChange={context.onBetSizesChange}
        />
        <CompactField
          label="Raise %"
          value={context.raiseSizes}
          onChange={context.onRaiseSizesChange}
        />
      </div>
    </div>
  );
}

function RangeSurface({ context }: { context: WorkspaceContext }) {
  const weights = context.rangeTab === 'oop' ? context.oop : context.ip;
  const onChange =
    context.rangeTab === 'oop' ? context.onOopChange : context.onIpChange;

  function selectAll() {
    const all: Record<string, number> = {};
    for (let row = 0; row < 13; row++) {
      for (let column = 0; column < 13; column++) {
        all[handClassLabel(row, column)] = 1;
      }
    }
    onChange(all);
  }

  return (
    <section
      aria-label="Range editor"
      className="hidden min-h-0 flex-col gap-2 rounded-lg border border-border bg-surface p-3 lg:flex"
    >
      <PlayerTabs value={context.rangeTab} onChange={context.setRangeTab} />
      <div className="solver-workspace-range-matrix mx-auto min-h-0 w-full">
        <RangeEditor
          weights={weights}
          onChange={onChange}
          title={`${context.rangeTab.toUpperCase()} range`}
          accent={playerColor(context.rangeTab)}
          compact
          showActions={false}
        />
        <div
          className="mt-2 grid grid-cols-3 gap-2 border-t border-border pt-2"
          aria-label="Range and spot tools"
        >
          <button
            type="button"
            onClick={selectAll}
            className="min-h-11 rounded-md border border-border px-3 text-xs font-semibold text-muted transition-colors hover:border-accent hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            All
          </button>
          <button
            type="button"
            onClick={() => onChange({})}
            className="min-h-11 rounded-md border border-border px-3 text-xs font-semibold text-muted transition-colors hover:border-accent hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            Clear
          </button>
          <button
            type="button"
            onClick={context.onClear}
            aria-label="Clear entire spot"
            title="Clear entire spot"
            className="inline-flex min-h-11 items-center justify-center gap-1.5 rounded-md border border-border px-3 text-xs font-semibold text-muted transition-colors hover:border-raise/60 hover:text-raise focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-raise"
          >
            <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
            Spot
          </button>
        </div>
      </div>
    </section>
  );
}

function EmptyStrategy({ context }: { context: WorkspaceContext }) {
  const message = context.solverError
    ? context.solverError
    : context.running
      ? 'Solving'
      : !context.available
        ? 'Starting solver'
        : context.missing ?? 'Ready';

  return (
    <div
      role={context.solverError ? 'alert' : 'status'}
      className="grid min-h-0 flex-1 place-items-center rounded-lg border border-dashed border-border bg-surface-2/50 p-6 text-center"
    >
      <div className="flex flex-col items-center gap-3 text-sm text-muted">
        {context.running || !context.available ? (
          <LoaderCircle className="h-5 w-5 animate-spin" aria-hidden="true" />
        ) : (
          <CircleDot className="h-5 w-5" aria-hidden="true" />
        )}
        <span>{message}</span>
      </div>
    </div>
  );
}

function StrategySurface({ context }: { context: WorkspaceContext }) {
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const node = context.result
    ? context.strategyTab === 'oop'
      ? context.result.oop
      : context.result.ip
    : null;

  return (
    <section
      aria-label="Solved strategy"
      className="flex min-h-0 flex-col gap-2 overflow-hidden rounded-lg border border-border bg-surface p-3"
    >
      <div className="flex shrink-0 items-center gap-2">
        <div className="min-w-0 flex-1">
          <PlayerTabs
            value={context.strategyTab}
            onChange={(player) => {
              context.setStrategyTab(player);
              setAnalysisOpen(false);
            }}
            suffix=" strategy"
          />
        </div>
        <button
          type="button"
          onClick={() => setAnalysisOpen((open) => !open)}
          aria-label="AI analysis"
          aria-pressed={analysisOpen}
          title="AI analysis"
          className={`grid h-11 w-11 shrink-0 place-items-center rounded-md border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
            analysisOpen
              ? 'border-accent bg-surface-2'
              : 'border-border hover:border-accent'
          }`}
        >
          <GeminiMark className="h-5 w-5" />
        </button>
      </div>
      {analysisOpen ? (
        <div className="min-h-0 flex-1 overflow-hidden rounded-md bg-surface-2/40 p-3">
          <AiPanel getSpot={context.getAnalysisSpot} embedded />
        </div>
      ) : node && context.result && !context.result.error ? (
        <div className="flex min-h-0 flex-1 flex-col justify-center overflow-hidden">
          <StrategyView
            node={node}
            framed={false}
            compact
            matrixClassName="solver-workspace-matrix"
          />
          {context.result.truncated && (
            <span className="mt-1 text-center text-[10px] font-medium uppercase text-muted">
              Range capped
            </span>
          )}
        </div>
      ) : (
        <EmptyStrategy context={context} />
      )}
      {!analysisOpen &&
        context.showSolverStats &&
        context.result &&
        !context.result.error && (
          <div className="shrink-0">
            <SolverNerdStats result={context.result} compact />
          </div>
        )}
    </section>
  );
}

export function SolverWorkspace(props: SolverWorkspaceProps) {
  const [rangeTab, setRangeTab] = useState<Player>('oop');
  const [strategyTab, setStrategyTab] = useState<Player>('oop');
  const context: WorkspaceContext = {
    ...props,
    rangeTab,
    setRangeTab,
    strategyTab,
    setStrategyTab,
  };

  return (
    <section
      data-solver-workspace
      className="solver-workspace grid h-[calc(100dvh-17rem)] grid-rows-[auto_minmax(0,1fr)] gap-2 overflow-hidden rounded-lg border border-border bg-bg p-2 text-fg md:h-[calc(100dvh-7.125rem)]"
    >
      <h1 className="sr-only">Postflop solver</h1>
      <header className="min-w-0 rounded-lg border border-border bg-surface px-3 py-2">
        <BoardControls context={context} />
      </header>
      <div className="grid min-h-0 gap-2 lg:grid-cols-[minmax(470px,0.95fr)_minmax(0,1.45fr)]">
        <RangeSurface context={context} />
        <StrategySurface context={context} />
      </div>
    </section>
  );
}
