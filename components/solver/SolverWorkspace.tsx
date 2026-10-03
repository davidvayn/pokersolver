'use client';

import { useEffect, useId, useRef, useState } from 'react';
import { Check, ChevronDown, LoaderCircle, Trash2 } from 'lucide-react';
import { AiPanel } from '@/components/ai/AiPanel';
import { GeminiMark } from '@/components/ai/GeminiMark';
import { CardSlots } from '@/components/board/CardPicker';
import { RangeEditor } from '@/components/range/RangeEditor';
import {
  SolverNerdStats,
  StrategyView,
} from '@/components/solver/SolverResults';
import {
  handClassLabel,
  rangeComboCount,
  weightsToRange,
  type Card,
} from '@/lib/cards';
import type { SpotContext } from '@/lib/ai/prompt';
import type { NodeStrategy, SolverResult } from '@/lib/solver/client';

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
  rangeOpen: boolean;
  setRangeOpen: (open: boolean) => void;
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
    <div
      className="flex items-center gap-4 border-b border-border"
      role="group"
    >
      {(['oop', 'ip'] as const).map((player) => (
        <button
          key={player}
          type="button"
          onClick={() => onChange(player)}
          aria-pressed={value === player}
          className={`min-h-11 border-b-2 px-1 text-xs font-semibold uppercase transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
            value === player
              ? 'border-accent text-fg'
              : 'border-transparent text-muted hover:text-fg'
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
  unit,
}: {
  label: string;
  value: string | number;
  onChange: (value: string) => void;
  number?: boolean;
  unit: 'bb' | '% pot';
}) {
  const id = useId();

  return (
    <label
      htmlFor={id}
      className="flex min-h-11 min-w-0 flex-col justify-center gap-1 border-b border-border focus-within:border-accent"
    >
      <span className="shrink-0 whitespace-nowrap text-xs font-medium text-muted">
        {label}
      </span>
      <span className="flex min-w-0 items-baseline gap-1">
        <input
          id={id}
          aria-label={`${label} (${unit})`}
          type={number ? 'number' : 'text'}
          inputMode={number ? 'decimal' : 'text'}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          className="w-full min-w-0 bg-transparent text-sm font-semibold tabular-nums text-fg outline-none"
        />
        <span className="shrink-0 text-xs text-muted" aria-hidden="true">
          {unit === '% pot' ? '%' : unit}
        </span>
      </span>
    </label>
  );
}

function BoardControls({ context }: { context: WorkspaceContext }) {
  return (
    <section
      aria-label="Solver settings"
      className="solver-settings min-w-0 border-b border-border pb-3"
    >
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <div className="solver-board relative flex shrink-0 items-center gap-3">
          <span className="text-xs font-medium text-muted">Board</span>
          <CardSlots
            count={5}
            cards={context.board}
            used={context.used}
            onChange={context.onBoardChange}
            size="md"
          />
        </div>
        <div className="grid min-w-0 flex-1 basis-[340px] grid-cols-4 gap-x-3">
          <CompactField
            label="Pot"
            unit="bb"
            value={context.pot}
            number
            onChange={(value) => context.onPotChange(parseFloat(value) || 0)}
          />
          <CompactField
            label="Stack"
            unit="bb"
            value={context.stack}
            number
            onChange={(value) => context.onStackChange(parseFloat(value) || 0)}
          />
          <CompactField
            label="Bet sizes"
            unit="% pot"
            value={context.betSizes}
            onChange={context.onBetSizesChange}
          />
          <CompactField
            label="Raise sizes"
            unit="% pot"
            value={context.raiseSizes}
            onChange={context.onRaiseSizesChange}
          />
        </div>
      </div>
    </section>
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
      id="solver-range-editor"
      aria-label="Range editor"
      className="min-w-0 border-b border-border bg-surface py-3"
    >
      <div className="mb-3 flex items-center justify-between gap-3">
        <PlayerTabs
          value={context.rangeTab}
          onChange={context.setRangeTab}
          suffix=" range"
        />
        <button
          type="button"
          onClick={() => {
            context.setRangeOpen(false);
            document
              .querySelector<HTMLButtonElement>(
                `[aria-label="Edit ${context.rangeTab.toUpperCase()} range"]`
              )
              ?.focus();
          }}
          className="inline-flex min-h-11 items-center gap-2 px-3 text-sm font-semibold text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          <Check className="h-4 w-4" aria-hidden="true" /> Done
        </button>
      </div>
      <div className="mx-auto w-full max-w-[440px]">
        <RangeEditor
          weights={weights}
          onChange={onChange}
          title={`${context.rangeTab.toUpperCase()} range`}
          accent={playerColor(context.rangeTab)}
          compact
          showActions={false}
        />
        <div
          className="mt-4 grid grid-cols-3 gap-2"
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

const EMPTY_NODES: Record<Player, NodeStrategy> = {
  oop: { title: 'OOP — first to act', actions: [], rows: [] },
  ip: { title: 'IP — vs check', actions: [], rows: [] },
};

function StrategySurface({ context }: { context: WorkspaceContext }) {
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const analysisTrigger = useRef<HTMLButtonElement>(null);
  const analysisPanel = useRef<HTMLElement>(null);

  useEffect(() => {
    if (analysisOpen) {
      analysisPanel.current
        ?.querySelector<HTMLButtonElement>('button')
        ?.focus();
    }
  }, [analysisOpen]);

  function closeAnalysis() {
    setAnalysisOpen(false);
    analysisTrigger.current?.focus();
  }
  const node =
    context.result && !context.result.error
      ? context.strategyTab === 'oop'
        ? context.result.oop
        : context.result.ip
      : EMPTY_NODES[context.strategyTab];
  const pending =
    context.running || !context.result || Boolean(context.result.error);
  const solving = !context.missing && !context.solverError && !context.result?.error &&
    (context.running || !context.result);

  return (
    <section
      aria-label="Solved strategy"
      className="flex min-w-0 flex-col gap-3"
    >
      <div className="relative flex shrink-0 flex-wrap items-center gap-x-4 gap-y-1">
        <div className="min-w-0">
          <PlayerTabs
            value={context.strategyTab}
            onChange={(player) => {
              context.setStrategyTab(player);
              context.setRangeOpen(false);
            }}
            suffix=" strategy"
          />
        </div>
        <div className="order-3 basis-full sm:order-none sm:ml-auto sm:basis-auto">
          <RangeSummaries context={context} />
        </div>
        <button
          ref={analysisTrigger}
          type="button"
          onClick={() => setAnalysisOpen((open) => !open)}
          aria-label="AI analysis"
          aria-pressed={analysisOpen}
          aria-expanded={analysisOpen}
          aria-controls="solver-ai-analysis"
          title="AI analysis"
          className={`ml-auto grid h-11 w-11 shrink-0 place-items-center rounded sm:ml-0 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
            analysisOpen
              ? 'bg-surface-2 text-accent'
              : 'text-muted hover:text-accent'
          }`}
        >
          <GeminiMark className="h-5 w-5" />
        </button>
        {solving && <div className="solver-progress absolute inset-x-0 -bottom-1" aria-hidden="true" />}
      </div>
      {context.rangeOpen && <RangeSurface context={context} />}
      <aside
        ref={analysisPanel}
        id="solver-ai-analysis"
        aria-label="AI analysis panel"
        hidden={!analysisOpen}
        className="solver-ai-panel fixed z-30 overflow-hidden rounded-lg border border-border bg-surface p-4 shadow-card"
        onKeyDown={(event) => {
          if (event.key === 'Escape') {
            event.preventDefault();
            event.stopPropagation();
            closeAnalysis();
          }
        }}
      >
        <AiPanel
          getSpot={context.getAnalysisSpot}
          embedded
          onClose={closeAnalysis}
        />
      </aside>
      <div className="flex min-w-0 flex-col">
        <StrategyView
          key={context.strategyTab}
          node={node}
          board={context.board}
          framed={false}
          compact
          matrixClassName="solver-workspace-matrix"
          settings={<BoardControls context={context} />}
          pending={pending}
        />
        {!pending && context.result?.truncated && (
          <span className="mt-1 text-center text-[10px] font-medium uppercase text-muted">
            Range capped
          </span>
        )}
      </div>
      {context.showSolverStats && context.result && !context.result.error && (
        <div className="shrink-0">
          <SolverNerdStats result={context.result} compact />
        </div>
      )}
    </section>
  );
}

function RangeSummaries({ context }: { context: WorkspaceContext }) {
  const error = context.solverError || context.result?.error;
  const solving = !error && !context.missing && (context.running || !context.result);
  return (
    <div className="flex flex-wrap items-center gap-x-5">
      {(['oop', 'ip'] as const).map((player) => {
        const count = rangeComboCount(weightsToRange(context[player]));
        return (
          <button
            key={player}
            type="button"
            aria-label={`Edit ${player.toUpperCase()} range`}
            aria-expanded={context.rangeOpen && context.rangeTab === player}
            aria-controls="solver-range-editor"
            onClick={() => {
              context.setRangeTab(player);
              context.setRangeOpen(
                !context.rangeOpen || context.rangeTab !== player
              );
            }}
            className="inline-flex min-h-11 items-center gap-2 text-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            <span
              className="h-2 w-2 rounded-full"
              style={{ background: playerColor(player) }}
              aria-hidden="true"
            />
            <span className="font-semibold">{player.toUpperCase()} range</span>
            <span className="text-muted">
              {count.toFixed(0)}
              <span className="sr-only sm:not-sr-only"> combos</span>
            </span>
            <ChevronDown
              className={`h-3.5 w-3.5 text-muted ${context.rangeOpen && context.rangeTab === player ? 'rotate-180' : ''}`}
              aria-hidden="true"
            />
          </button>
        );
      })}
      <span
        role={error ? 'alert' : 'status'}
        aria-live="polite"
        className={`ml-auto inline-flex items-center gap-2 ${solving ? 'text-sm font-semibold text-accent' : 'text-[10px] text-muted'}`}
        title={!solving && !error && !context.missing ? 'Strategy ready' : undefined}
      >
        {error || context.missing || (solving ? (
          <>
            <LoaderCircle className="h-4 w-4 motion-safe:animate-spin" aria-hidden="true" />
            {context.available ? 'Solving…' : 'Starting solver…'}
          </>
        ) : (
          <>
            <Check className="h-3 w-3" aria-hidden="true" />
            <span className="sr-only">Solved</span>
          </>
        ))}
      </span>
    </div>
  );
}

export function SolverWorkspace(props: SolverWorkspaceProps) {
  const [rangeOpen, setRangeOpen] = useState(
    () =>
      !Object.values(props.oop).some(Boolean) ||
      !Object.values(props.ip).some(Boolean)
  );
  const [rangeTab, setRangeTab] = useState<Player>('oop');
  const [strategyTab, setStrategyTab] = useState<Player>('oop');
  const context: WorkspaceContext = {
    ...props,
    rangeOpen,
    setRangeOpen,
    rangeTab,
    setRangeTab,
    strategyTab,
    setStrategyTab,
  };

  return (
    <section data-solver-workspace className="solver-workspace min-w-0 text-fg">
      <h1 className="sr-only">Postflop solver</h1>
      <StrategySurface context={context} />
    </section>
  );
}
