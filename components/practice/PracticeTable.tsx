import {
  AlertTriangle,
  BarChart3,
  CircleDollarSign,
  LoaderCircle,
  RefreshCw,
  ShieldAlert,
} from 'lucide-react';
import { PokerCard } from '@/components/cards/PokerCard';
import { totalPotBb } from '@/lib/practice-engine';
import { practiceActionChoices } from '@/lib/practice-grading';
import type {
  HandState,
  LegalAction,
  PolicyNode,
  PracticeMode,
  Seat,
} from '@/lib/practice-types';

type TableStatus =
  | 'loading'
  | 'transitioning'
  | 'solving'
  | 'unavailable'
  | 'decision'
  | 'feedback'
  | 'review'
  | 'error';

interface PracticeTableProps {
  state: HandState | null;
  node: PolicyNode | null;
  status: TableStatus;
  mode: PracticeMode;
  unavailableMessage?: string;
  errorMessage?: string;
  revealOpponent: boolean;
  selectedActionId: string | null;
  onAction: (action: LegalAction) => void;
  onContinue: () => void;
  onRetry: () => void;
  onOpenAnalyst?: () => void;
}

function SeatDisplay({
  seat,
  state,
  opponent,
  revealOpponent,
}: {
  seat: Seat;
  state: HandState;
  opponent: boolean;
  revealOpponent: boolean;
}) {
  const cards = state.holeCards[seat];
  const active = state.toAct === seat;
  const label = seat === 'button-small-blind' ? 'BTN / SB' : 'Big blind';
  return (
    <div
      className={`practice-seat ${active ? 'practice-seat-active' : ''}`}
      aria-label={`${opponent ? 'Opponent' : 'Hero'}, ${label}, ${state.stacksBb[seat].toFixed(1)} big blinds`}
    >
      <div className="practice-seat-meta">
        <span className="practice-seat-label">
          {opponent ? 'Opponent' : 'Hero'} · {label}
        </span>
        {seat === state.button && (
          <span
            className="grid h-6 w-6 place-items-center rounded-full bg-white text-[10px] font-black text-neutral-900 shadow"
            title="Dealer button"
            aria-label="Dealer button"
          >
            D
          </span>
        )}
      </div>
      <div className="practice-seat-body">
        <div className="practice-card-hand">
          <PokerCard card={cards[0]} hidden={opponent && !revealOpponent} />
          <PokerCard card={cards[1]} hidden={opponent && !revealOpponent} />
        </div>
        <span className="practice-stack">
          {state.stacksBb[seat].toFixed(1)}bb
        </span>
      </div>
    </div>
  );
}

function modeLabel(mode: PracticeMode): string {
  switch (mode) {
    case 'full-hand':
      return 'Full hand';
    case 'preflop':
      return 'Preflop only';
    case 'postflop':
      return 'Postflop only';
    case 'push-fold':
      return 'Push/fold';
  }
}

export function PracticeTable({
  state,
  node,
  status,
  mode,
  unavailableMessage,
  errorMessage,
  revealOpponent,
  selectedActionId,
  onAction,
  onContinue,
  onRetry,
  onOpenAnalyst,
}: PracticeTableProps) {
  const opponent: Seat = state
    ? state.hero === 'button-small-blind'
      ? 'big-blind'
      : 'button-small-blind'
    : 'big-blind';
  const pot = state
    ? state.terminal
      ? state.result?.potBb ?? 0
      : totalPotBb(state)
    : 1.5;
  const liveMessage =
    status === 'loading'
      ? 'Loading practice spot'
      : status === 'transitioning'
        ? state?.street === 'preflop'
          ? 'Completing the preflop action.'
          : 'The flop is dealt. Preparing the first postflop decision.'
        : status === 'solving'
          ? 'Currently solving the postflop strategy. Almost done.'
          : status === 'decision'
            ? `Action on hero, ${state?.street ?? 'preflop'}`
            : status === 'feedback'
              ? 'Decision reviewed. Continue the hand when ready.'
              : status === 'review'
                ? 'Hand review complete. Continue when ready.'
                : status === 'unavailable'
                  ? 'Practice model unavailable'
                  : errorMessage ?? 'Practice error';

  return (
    <section aria-labelledby="practice-table-title" className="min-w-0">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 id="practice-table-title" className="text-xl font-semibold sm:text-2xl">
            Heads-up table
          </h1>
          <p className="mt-1 text-sm text-muted">
            {modeLabel(mode)} · {state?.depthBb ?? 20}bb · 0.5/1bb · no rake
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {onOpenAnalyst && (
            <button
              type="button"
              onClick={onOpenAnalyst}
              className="inline-flex min-h-11 items-center gap-2 rounded-full border border-border bg-surface px-3 text-xs font-semibold shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent xl:hidden"
            >
              <BarChart3 className="h-4 w-4 text-accent" aria-hidden="true" />
              Analyst
            </button>
          )}
        </div>
      </div>

      <div className="practice-table-shell">
        <div className="practice-felt" aria-describedby="table-status">
          {state ? (
            <>
              <div className="practice-opponent-seat">
                <SeatDisplay
                  seat={opponent}
                  state={state}
                  opponent
                  revealOpponent={revealOpponent}
                />
              </div>

              <div className="practice-board" aria-label="Community cards">
                {[0, 1, 2, 3, 4].map((index) => (
                  <PokerCard key={index} card={state.board[index]} />
                ))}
              </div>

              <div className="practice-pot" aria-label={`Pot ${pot.toFixed(1)} big blinds`}>
                <CircleDollarSign className="h-4 w-4" aria-hidden="true" />
                <span className="font-mono font-semibold">{pot.toFixed(1)}bb</span>
              </div>

              <div className="practice-hero-seat">
                <SeatDisplay
                  seat={state.hero}
                  state={state}
                  opponent={false}
                  revealOpponent
                />
              </div>

              <ol className="practice-action-strip" aria-label="Action history">
                {state.actionHistory.slice(-4).map((action) => (
                  <li key={action.id}>
                    <span className="text-white/55">
                      {action.actor === state.hero ? 'You' : 'Villain'}
                    </span>{' '}
                    {action.label}
                  </li>
                ))}
                {state.actionHistory.length === 0 && <li>Blinds posted</li>}
              </ol>
            </>
          ) : (
            <div className="practice-empty-table" aria-hidden="true">
              <div className="practice-seat practice-seat-ghost">
                <div className="h-3 w-28 rounded bg-white/10" />
                <div className="mt-3 flex gap-2">
                  <PokerCard hidden />
                  <PokerCard hidden />
                </div>
              </div>
              <div className="practice-board">
                {[0, 1, 2, 3, 4].map((index) => (
                  <PokerCard key={index} />
                ))}
              </div>
              <div className="practice-seat practice-seat-ghost">
                <div className="h-3 w-20 rounded bg-white/10" />
                <div className="mt-3 flex gap-2">
                  <PokerCard hidden />
                  <PokerCard hidden />
                </div>
              </div>
            </div>
          )}

          {(status === 'loading' ||
            status === 'solving' ||
            status === 'unavailable' ||
            status === 'error') && (
            <div className="practice-table-overlay">
              <div className="max-w-md rounded-lg border border-white/15 bg-neutral-950/85 p-5 text-center text-white shadow-2xl backdrop-blur-sm">
                {status === 'loading' || status === 'solving' ? (
                  <LoaderCircle className="mx-auto h-6 w-6 animate-spin motion-reduce:animate-none" aria-hidden="true" />
                ) : status === 'unavailable' ? (
                  <ShieldAlert className="mx-auto h-6 w-6 text-amber-300" aria-hidden="true" />
                ) : (
                  <AlertTriangle className="mx-auto h-6 w-6 text-red-300" aria-hidden="true" />
                )}
                <p className="mt-3 text-sm font-semibold">
                  {status === 'loading'
                    ? 'Preparing the table'
                    : status === 'solving'
                      ? 'Currently solving'
                      : status === 'unavailable'
                        ? 'Validated model not available'
                        : 'The table is paused'}
                </p>
                <p className="mt-1 text-xs leading-5 text-white/65">
                  {status === 'loading'
                    ? 'Dealing unique cards and pinning the model version.'
                    : status === 'solving'
                      ? 'Almost done — the postflop strategy started during preflop.'
                      : status === 'unavailable'
                        ? unavailableMessage
                        : errorMessage}
                </p>
                {status !== 'loading' && status !== 'solving' && (
                  <button
                    type="button"
                    onClick={onRetry}
                    className="mt-4 inline-flex min-h-11 items-center gap-2 rounded-md bg-white px-4 text-sm font-semibold text-neutral-950 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                  >
                    <RefreshCw className="h-4 w-4" aria-hidden="true" />
                    Retry
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      <p id="table-status" className="sr-only" role="status" aria-live="polite">
        {liveMessage}
      </p>

      <div className="practice-action-dock">
        {status === 'decision' && node ? (
          <div className="grid w-full grid-cols-2 gap-2 sm:flex sm:justify-center">
            {practiceActionChoices(node.actions).map((action) => (
              <button
                key={action.id}
                type="button"
                onClick={() => onAction(action)}
                disabled={selectedActionId !== null}
                className={`practice-action-button practice-action-${action.kind} ${selectedActionId === action.id ? 'ring-2 ring-accent ring-offset-2 ring-offset-bg' : ''}`}
              >
                <span>{action.label}</span>
              </button>
            ))}
          </div>
        ) : status === 'feedback' || status === 'review' ? (
          <button
            type="button"
            onClick={onContinue}
            className="min-h-12 w-full rounded-md bg-accent px-5 text-sm font-semibold text-accent-fg shadow-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent sm:w-auto sm:min-w-48"
          >
            {status === 'feedback' ? 'Continue hand' : 'Next hand'}
          </button>
        ) : (
          <p className="py-3 text-center text-sm text-muted">
            {status === 'loading'
              ? 'Loading…'
              : status === 'transitioning'
                ? state?.street === 'preflop'
                  ? 'Completing preflop…'
                  : 'Flop dealt · preparing actions…'
                : status === 'solving'
                  ? 'Currently solving…'
                  : status === 'unavailable'
                    ? 'Choose an available mode in Settings.'
                    : 'Actions are paused.'}
          </p>
        )}
      </div>
    </section>
  );
}
