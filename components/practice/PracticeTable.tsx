import {
  AlertTriangle,
  BadgeCheck,
  CircleDollarSign,
  LoaderCircle,
  RefreshCw,
  ShieldAlert,
} from 'lucide-react';
import { cardRank, cardSuit, RANKS, type Card } from '@/lib/cards';
import { totalPotBb } from '@/lib/practice-engine';
import type {
  HandState,
  LegalAction,
  PolicyNode,
  PracticeMode,
  Seat,
} from '@/lib/practice-types';

type TableStatus =
  | 'loading'
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
  modelLabel: string;
  unavailableMessage?: string;
  errorMessage?: string;
  revealOpponent: boolean;
  selectedActionId: string | null;
  onAction: (action: LegalAction) => void;
  onContinue: () => void;
  onRetry: () => void;
}

const SUIT_GLYPHS = ['♣', '♦', '♥', '♠'] as const;

function CardView({ card, hidden = false }: { card?: Card; hidden?: boolean }) {
  if (hidden) {
    return (
      <span
        className="playing-card playing-card-back"
        aria-label="Face-down card"
      >
        <span aria-hidden="true">PL</span>
      </span>
    );
  }
  if (card === undefined) {
    return <span className="playing-card playing-card-empty" aria-hidden="true" />;
  }
  const rank = RANKS[cardRank(card)];
  const suit = SUIT_GLYPHS[cardSuit(card)];
  const red = cardSuit(card) === 1 || cardSuit(card) === 2;
  return (
    <span
      className={`playing-card ${red ? 'playing-card-red' : ''}`}
      aria-label={`${rank} ${suit}`}
    >
      <span>{rank}</span>
      <span aria-hidden="true">{suit}</span>
    </span>
  );
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
      <div className="flex items-center justify-between gap-3">
        <span className="text-[11px] font-semibold uppercase tracking-wide text-white/65">
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
      <div className="mt-2 flex items-end justify-between gap-3">
        <div className="flex gap-1.5">
          <CardView card={cards[0]} hidden={opponent && !revealOpponent} />
          <CardView card={cards[1]} hidden={opponent && !revealOpponent} />
        </div>
        <span className="font-mono text-sm font-semibold text-white">
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
  modelLabel,
  unavailableMessage,
  errorMessage,
  revealOpponent,
  selectedActionId,
  onAction,
  onContinue,
  onRetry,
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
        <div className="flex items-center gap-2 rounded-full border border-border bg-surface px-3 py-1.5 text-xs">
          <BadgeCheck className="h-4 w-4 text-accent" aria-hidden="true" />
          <span className="font-medium">{modelLabel}</span>
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
                  <CardView key={index} card={state.board[index]} />
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
                  <CardView hidden />
                  <CardView hidden />
                </div>
              </div>
              <div className="practice-board">
                {[0, 1, 2, 3, 4].map((index) => (
                  <CardView key={index} />
                ))}
              </div>
              <div className="practice-seat practice-seat-ghost">
                <div className="h-3 w-20 rounded bg-white/10" />
                <div className="mt-3 flex gap-2">
                  <CardView hidden />
                  <CardView hidden />
                </div>
              </div>
            </div>
          )}

          {(status === 'loading' || status === 'unavailable' || status === 'error') && (
            <div className="practice-table-overlay">
              <div className="max-w-md rounded-lg border border-white/15 bg-neutral-950/85 p-5 text-center text-white shadow-2xl backdrop-blur-sm">
                {status === 'loading' ? (
                  <LoaderCircle className="mx-auto h-6 w-6 animate-spin motion-reduce:animate-none" aria-hidden="true" />
                ) : status === 'unavailable' ? (
                  <ShieldAlert className="mx-auto h-6 w-6 text-amber-300" aria-hidden="true" />
                ) : (
                  <AlertTriangle className="mx-auto h-6 w-6 text-red-300" aria-hidden="true" />
                )}
                <p className="mt-3 text-sm font-semibold">
                  {status === 'loading'
                    ? 'Preparing the table'
                    : status === 'unavailable'
                      ? 'Validated model not available'
                      : 'The table is paused'}
                </p>
                <p className="mt-1 text-xs leading-5 text-white/65">
                  {status === 'loading'
                    ? 'Dealing unique cards and pinning the model version.'
                    : status === 'unavailable'
                      ? unavailableMessage
                      : errorMessage}
                </p>
                {status !== 'loading' && (
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
            {node.actions.map((action) => (
              <button
                key={action.id}
                type="button"
                onClick={() => onAction(action)}
                disabled={selectedActionId !== null}
                className={`practice-action-button practice-action-${action.kind} ${selectedActionId === action.id ? 'ring-2 ring-accent ring-offset-2 ring-offset-bg' : ''}`}
              >
                <span>{action.label}</span>
                <span className="font-mono text-xs opacity-75">
                  {(action.probability * 100).toFixed(action.probability > 0.995 ? 1 : 0)}%
                </span>
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
              : status === 'unavailable'
                ? 'Choose an available mode in Settings.'
                : 'Actions are paused.'}
          </p>
        )}
      </div>
    </section>
  );
}
