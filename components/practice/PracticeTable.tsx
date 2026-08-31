'use client';

import { useState, type CSSProperties } from 'react';
import {
  AlertTriangle,
  BarChart3,
  Check,
  CircleDollarSign,
  Coins,
  History,
  LoaderCircle,
  LogOut,
  RefreshCw,
  ShieldAlert,
  Volume2,
  VolumeX,
} from 'lucide-react';
import { PokerCard } from '@/components/cards/PokerCard';
import { usePracticeTableSounds } from '@/components/practice/usePracticeTableSounds';
import {
  playPracticeSound,
  unlockPracticeAudio,
} from '@/lib/practice-sounds';
import { totalPotBb } from '@/lib/practice-engine';
import { practiceActionChoices } from '@/lib/practice-grading';
import type {
  HandState,
  LegalAction,
  PolicyNode,
  PublicAction,
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
  folding,
}: {
  seat: Seat;
  state: HandState;
  opponent: boolean;
  revealOpponent: boolean;
  folding: boolean;
}) {
  const cards = state.holeCards[seat];
  const active = state.toAct === seat;
  const label = seat === 'button-small-blind' ? 'BTN / SB' : 'Big blind';
  const owner = opponent ? 'Opponent' : 'Hero';
  return (
    <div
      className={`practice-seat ${active ? 'practice-seat-active' : ''} ${folding ? 'practice-seat-folding' : ''}`}
      role="group"
      aria-label={`${owner}, ${label}`}
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
        <VirtualChipStack
          amountBb={state.stacksBb[seat]}
          label={`${owner} stack`}
          variant="player"
        />
      </div>
    </div>
  );
}

function ChipPile({ secondary = false }: { secondary?: boolean }) {
  return (
    <span
      className={`practice-chip-pile ${secondary ? 'practice-chip-pile-secondary' : ''}`}
    >
      {[0, 1, 2, 3].map((level) => (
        <i
          key={level}
          className="practice-virtual-chip"
          style={{ '--practice-chip-level': level } as CSSProperties}
        />
      ))}
    </span>
  );
}

function VirtualChipStack({
  amountBb,
  label,
  variant,
}: {
  amountBb: number;
  label: string;
  variant: 'player' | 'pot';
}) {
  return (
    <div
      key={`${variant}-${amountBb.toFixed(2)}`}
      className={`practice-money-stack practice-money-stack-${variant}`}
      role="img"
      aria-label={`${label}, ${amountBb.toFixed(1)} big blinds`}
    >
      <span className="practice-chip-collection" aria-hidden="true">
        <ChipPile />
        {variant === 'pot' && <ChipPile secondary />}
      </span>
      <span className="practice-money-copy" aria-hidden="true">
        <small>{variant === 'pot' ? 'Pot' : 'Stack'}</small>
        <strong>{amountBb.toFixed(1)}bb</strong>
      </span>
    </div>
  );
}

function ActionEventIcon({ action }: { action: PublicAction | null }) {
  if (!action) {
    return <CircleDollarSign aria-hidden="true" />;
  }
  switch (action.kind) {
    case 'fold':
      return <LogOut aria-hidden="true" />;
    case 'check':
      return <Check aria-hidden="true" />;
    case 'call':
      return <CircleDollarSign aria-hidden="true" />;
    case 'bet':
    case 'raise':
    case 'all-in':
      return <Coins aria-hidden="true" />;
  }
}

function isWagerAction(action: PublicAction | null): boolean {
  return Boolean(
    action && ['call', 'bet', 'raise', 'all-in'].includes(action.kind)
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
  const [soundsEnabled, setSoundsEnabled] = useState(true);
  usePracticeTableSounds(state, soundsEnabled);

  function toggleSounds() {
    const next = !soundsEnabled;
    setSoundsEnabled(next);
    if (next) {
      void unlockPracticeAudio().then((unlocked) => {
        if (unlocked) playPracticeSound({ kind: 'chips' });
      });
    }
  }

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
  const latestAction = state?.actionHistory.at(-1) ?? null;
  const latestActor = latestAction
    ? latestAction.actor === state?.hero
      ? 'You'
      : 'Opponent'
    : null;
  const showBlindsEvent = Boolean(
    state &&
      state.actionHistory.length === 0 &&
      !['loading', 'unavailable', 'error'].includes(status)
  );
  const showActionEvent = Boolean(latestAction || showBlindsEvent);
  const eventKind = latestAction?.kind ?? 'blinds';
  const eventPosition = latestAction
    ? latestAction.actor === state?.hero
      ? 'hero'
      : 'opponent'
    : 'center';
  const actionAnnouncement = latestAction
    ? `${latestActor} ${latestAction.label}.`
    : showBlindsEvent
      ? 'Blinds posted.'
      : '';
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
            ? `${actionAnnouncement} Action on hero, ${state?.street ?? 'preflop'}`
            : status === 'feedback'
              ? `${actionAnnouncement} Decision reviewed. Continue the hand when ready.`
              : status === 'review'
                ? `${actionAnnouncement} Hand review complete. Continue when ready.`
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
          <button
            type="button"
            onClick={toggleSounds}
            aria-pressed={soundsEnabled}
            aria-label="Table sounds"
            title={soundsEnabled ? 'Mute table sounds' : 'Turn table sounds on'}
            className="grid h-11 w-11 place-items-center rounded-full border border-border bg-surface text-muted shadow-sm transition-colors hover:border-accent hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            {soundsEnabled ? (
              <Volume2 className="h-4 w-4" aria-hidden="true" />
            ) : (
              <VolumeX className="h-4 w-4" aria-hidden="true" />
            )}
          </button>
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
                  folding={latestAction?.kind === 'fold' && latestAction.actor === opponent}
                />
              </div>

              <div className="practice-board" aria-label="Community cards">
                {[0, 1, 2, 3, 4].map((index) => (
                  <span
                    key={`${index}-${state.board[index] ?? 'empty'}`}
                    className={`practice-board-card ${state.board[index] !== undefined ? 'practice-board-card-dealt' : ''}`}
                    style={{
                      '--practice-deal-order': index,
                    } as CSSProperties}
                  >
                    <PokerCard card={state.board[index]} />
                  </span>
                ))}
              </div>

              <div className="practice-pot">
                <VirtualChipStack amountBb={pot} label="Pot" variant="pot" />
              </div>

              <div className="practice-hero-seat">
                <SeatDisplay
                  seat={state.hero}
                  state={state}
                  opponent={false}
                  revealOpponent
                  folding={latestAction?.kind === 'fold' && latestAction.actor === state.hero}
                />
              </div>

              {showActionEvent && (
                <div
                  key={latestAction?.id ?? `${state.id}-blinds-${status}`}
                  className={`practice-action-event practice-action-event-${eventKind} practice-action-event-${eventPosition} ${isWagerAction(latestAction) ? 'practice-action-event-wager' : ''}`}
                  aria-hidden="true"
                >
                  <span className="practice-action-event-icon">
                    <ActionEventIcon action={latestAction} />
                  </span>
                  <span className="practice-action-event-copy">
                    <small>{latestActor ?? 'Table'}</small>
                    <strong>{latestAction?.label ?? 'Blinds posted'}</strong>
                  </span>
                  {isWagerAction(latestAction) && (
                    <span className="practice-action-chip-flight">
                      <i />
                      <i />
                      <i />
                    </span>
                  )}
                </div>
              )}

              <div className="practice-action-strip">
                <div className="practice-action-strip-heading" aria-hidden="true">
                  <History />
                  Table log
                </div>
                <ol aria-label="Action history">
                  {state.actionHistory.slice(-6).map((action, index, actions) => (
                    <li
                      key={action.id}
                      className={index === actions.length - 1 ? 'practice-action-log-latest' : ''}
                    >
                      <span className="practice-action-log-actor">
                        {action.actor === state.hero ? 'You' : 'Opponent'}
                      </span>
                      <strong>{action.label}</strong>
                    </li>
                  ))}
                  {state.actionHistory.length === 0 && (
                    <li className="practice-action-log-latest">
                      <span className="practice-action-log-actor">Table</span>
                      <strong>Blinds posted</strong>
                    </li>
                  )}
                </ol>
              </div>
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
                  <span className="practice-board-card" key={index}>
                    <PokerCard />
                  </span>
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
          <div className="flex w-full gap-2 sm:w-auto">
            {onOpenAnalyst && (
              <button
                type="button"
                onClick={onOpenAnalyst}
                className="inline-flex min-h-12 flex-1 items-center justify-center gap-2 rounded-md border border-border bg-surface px-3 text-sm font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent sm:min-w-44 xl:hidden"
              >
                <BarChart3 className="h-4 w-4 text-accent" aria-hidden="true" />
                {status === 'feedback' ? 'Review decision' : 'Review hand'}
              </button>
            )}
            <button
              type="button"
              onClick={onContinue}
              className="min-h-12 flex-1 rounded-md bg-accent px-4 text-sm font-semibold text-accent-fg shadow-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent sm:min-w-48"
            >
              {status === 'feedback' ? 'Continue hand' : 'Next hand'}
            </button>
          </div>
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
