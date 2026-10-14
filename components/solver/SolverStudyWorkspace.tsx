'use client';

import { useEffect, useRef, useState, type CSSProperties } from 'react';
import { ArrowLeft, Check, Info, LoaderCircle, RotateCcw } from 'lucide-react';
import { CardSlots } from '@/components/board/CardPicker';
import { AiPanel } from '@/components/ai/AiPanel';
import { GeminiMark } from '@/components/ai/GeminiMark';
import { StrategyView, colorForActions } from '@/components/solver/SolverResults';
import { useSolverStudy } from '@/components/solver/useSolverStudy';
import { cardToStr } from '@/lib/cards';
import { activeFullHandDepths } from '@/lib/practice-models';
import { totalPotBb } from '@/lib/practice-engine';
import { solverAmount } from '@/lib/solver/tree';
import type { NodeStrategy } from '@/lib/solver/client';
import type { SpotContext } from '@/lib/ai/prompt';

const streetLabel = (street: string) => street[0].toUpperCase() + street.slice(1);

export function SolverStudyWorkspace() {
  const study = useSolverStudy();
  const state = study.line?.state;
  const completed = Boolean(state?.terminal);
  const waiting = study.line?.waitingForCards;
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const activeCard = useRef<HTMLParagraphElement>(null);
  const status = useRef<HTMLParagraphElement>(null);
  const previousChoices = useRef(study.choices);
  useEffect(() => {
    if (previousChoices.current !== study.choices) (activeCard.current ?? status.current)?.focus({ preventScroll: true });
    activeCard.current?.closest('[data-solver-decision-card]')?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    previousChoices.current = study.choices;
  }, [study.choices, state?.street, state?.toAct]);
  const title = state?.terminal ? state.result?.reason === 'fold' ? `${state.result.winner === 'big-blind' ? 'OOP' : 'IP'} wins by fold` : 'Showdown · range review'
    : waiting ? `${Object.values(state?.stacksBb ?? {}).some((stack) => stack <= 1e-9) ? 'All-in runout' : 'Betting round complete'} · select ${waiting === 3 ? 'a flop' : waiting === 4 ? 'a turn card' : 'a river card'}`
    : `${streetLabel(state?.street ?? 'preflop')} · ${state?.toAct === 'button-small-blind' ? 'IP (BTN/SB)' : 'OOP (BB)'} to act`;
  const node: NodeStrategy = study.node ?? { title: study.target ? `${study.target.toAct === 'big-blind' ? 'OOP' : 'IP'} — ${study.target.street}` : 'Shared full-hand policy', actions: [], rows: [] };
  const availableDepths = activeFullHandDepths();
  const depths = [...new Set([...availableDepths, 50, 100])].sort((a, b) => a - b);
  const settings = <section aria-label="Solver settings" className="solver-settings min-w-0 border-b border-border pb-3">
    <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
      <div className="solver-board relative flex shrink-0 items-center gap-3">
        <span className="text-xs font-medium text-muted">Board</span>
        <CardSlots count={5} cards={study.board} used={new Set(study.board)} onChange={study.setBoard} />
      </div>
      <label className="flex min-h-11 min-w-0 flex-col gap-1 text-xs text-muted">Starting stack
        <select aria-label="Starting stack (bb)" value={study.depthBb} onChange={(e) => study.changeDepth(Number(e.target.value))}
          className="min-h-9 max-w-full bg-transparent font-semibold text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent">
          {depths.map((depth) => <option key={depth} value={depth} disabled={!availableDepths.includes(depth)}>{depth}bb{availableDepths.includes(depth) ? '' : ' · pilot, unavailable'}</option>)}
        </select>
      </label>
      <span className="text-xs text-muted">Policy-conditioned ranges · exact board cards</span>
    </div>
  </section>;
  function getAnalysisSpot(): SpotContext | null {
    if (!study.target || !study.node || study.pending || study.error) return null;
    return { kind: study.target.street === 'preflop' ? 'preflop' : 'postflop',
      description: `Shared full-hand ${study.target.street} policy conditioned on the complete public trajectory. Experimental; exploitability gate deferred. Available EVs have low confidence; unknown values are not estimates.`,
      hero: study.target.toAct === 'big-blind' ? 'OOP / BB' : 'IP / BTN/SB', villain: study.target.toAct === 'big-blind' ? 'IP / BTN/SB' : 'OOP / BB',
      board: study.target.board.map(cardToStr).join(''), potBB: totalPotBb(study.target), stackBB: Math.min(...Object.values(study.target.stacksBb)),
      solver: { truncated: false, nodes: [study.node] }, extra: { 'Model version': study.manifest?.version ?? '', 'Starting stack': `${study.depthBb}bb`,
        'Public actions': study.target.actionHistory.map((action) => `${action.street} ${action.actor} ${action.label}`).join(' → ') } };
  }
  return <section data-solver-workspace data-shared-multi-street-engine className="solver-workspace min-w-0 text-fg">
    <h1 className="sr-only">Multi-street heads-up solver</h1>
    <p className="sr-only" data-solver-rules>Home game · rake-free multi-street engine shared with Practice. Only installed policy depths are available.</p>
    <section aria-label="Betting line" aria-busy={study.pending} className="solver-line-navigator relative min-w-0 border-y border-border py-2">
      <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1">
        <div className="flex items-center gap-1">
          <button type="button" disabled={!study.choices.length} onClick={study.back} className="inline-flex min-h-11 items-center gap-1 rounded px-2 text-xs text-muted disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"><ArrowLeft className="h-3.5 w-3.5" aria-hidden />Back</button>
          <button type="button" disabled={!study.choices.length} aria-label="Reset betting line" onClick={study.reset} className="inline-flex min-h-11 items-center gap-1 rounded px-2 text-xs text-muted disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"><RotateCcw className="h-3.5 w-3.5" aria-hidden />Start</button>
          <details>
            <summary aria-label="Solver model details" className="flex min-h-11 cursor-pointer list-none items-center gap-1 rounded px-2 text-xs text-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"><Info className="h-3.5 w-3.5" aria-hidden />Model</summary>
            <div className="absolute left-0 top-14 z-30 w-[min(400px,calc(100vw-32px))] rounded border border-border bg-surface p-3 text-xs leading-5 text-muted shadow-card">
              <p>Home game · rake-free heads-up, 0.5/1bb blinds. The same full-hand engine and frozen resolver as Practice.</p>
              <p className="mt-2">{study.manifest?.version}. Experimental policy: full-game exploitability has not passed. One modeled postflop raise, plus all-in, across flop/turn/river. Input ranges are reconstructed from the entire action trajectory.</p>
              <p className="mt-2">EVs are approximate and low-confidence. Missing values remain unavailable; relative combo weights are scaled to the highest-reach combination. Public range review does not invent a private showdown winner.</p>
              <p className="mt-2">50bb and 100bb are unavailable until their new policies are trained and validated.</p>
            </div>
          </details>
        </div>
        <p ref={status} tabIndex={-1} role="status" aria-live="polite" className="min-w-0 text-xs font-semibold focus:outline-none">{title}</p>
        {state && <dl className="ml-auto flex flex-wrap gap-x-4 gap-y-1 text-xs tabular-nums">
          {([['Pot', state.result?.reason === 'fold' ? state.result.potBb : totalPotBb(state)], ['OOP', state.stacksBb['big-blind']], ['IP', state.stacksBb['button-small-blind']]] as const).map(([label, value]) => <div key={label} className="flex gap-1"><dt className="text-muted">{label}</dt><dd className="font-semibold">{solverAmount(value)}bb</dd></div>)}
        </dl>}
      </div>
      <ol aria-label="Decision cards" className="solver-decision-rail">
        {study.decisions.map((decision, index) => {
          const current = index === study.choices.length && !completed && !waiting;
          const colors = colorForActions(decision.actions.map((action) => action.label));
          const seat = decision.state.toAct!;
          return <li key={index} data-solver-decision-card data-current={current} className={`solver-decision-card ${current ? 'border-accent' : 'border-border'}`}>
            <p ref={current ? activeCard : undefined} tabIndex={-1} className={`flex min-h-9 items-center justify-between gap-2 border-b border-border px-2 text-xs focus:outline-none ${current ? 'font-bold text-accent' : 'font-semibold text-muted'}`}>
              <span>{seat === 'big-blind' ? 'OOP' : 'IP'} <span className="font-normal text-muted">· {streetLabel(decision.state.street)}</span></span><span className="font-normal text-muted">{solverAmount(decision.state.stacksBb[seat])}bb</span>
            </p>
            <div role="group" aria-label={`Modeled actions, ${seat === 'big-blind' ? 'OOP' : 'IP'}, ${decision.state.street}, decision ${index + 1}`} className="flex flex-col gap-1 p-1">
              {decision.actions.map((action, actionIndex) => {
                const pressed = !completed && decision.selected?.id === action.id;
                return <button key={action.id} type="button" aria-pressed={pressed}
                  disabled={index >= (study.line?.decisions.length ?? 0) || (current && (study.pending || Boolean(study.error)))}
                  data-solver-action-index={actionIndex} data-solver-decision-index={index} onClick={() => study.choose(index, action)}
                  style={{ '--solver-action-color': colors[action.label] } as CSSProperties}
                  className="solver-decision-option focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent">
                  <span>{action.label}</span>{pressed && <Check className="ml-auto h-3 w-3 shrink-0" aria-hidden />}
                </button>;
              })}
            </div>
          </li>;
        })}
      </ol>
    </section>
    <section aria-label="Solved strategy" className="mt-3 flex min-w-0 flex-col gap-3">
      <div className="relative flex min-h-11 flex-wrap items-center justify-between gap-2">
        <p className="py-3 text-xs font-semibold uppercase text-muted">{completed || waiting ? 'Last decision' : 'Current strategy'} · {node.title}</p>
        <span role="status" className="ml-auto inline-flex items-center gap-2 text-xs text-muted">{study.pending && !study.line?.unappliedChoices ? <><LoaderCircle className="h-4 w-4 motion-safe:animate-spin" aria-hidden />Solving…</> : study.error ? 'Table paused' : 'Shared full-hand policy'}</span>
        <button type="button" aria-label="AI analysis" aria-expanded={analysisOpen} aria-pressed={analysisOpen} onClick={() => setAnalysisOpen((open) => !open)} className="grid h-11 w-11 place-items-center rounded text-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"><GeminiMark className="h-5 w-5" /></button>
        {study.pending && <div className="solver-progress absolute inset-x-0 -bottom-1" aria-hidden />}
      </div>
      {study.error && <div role="alert" className="flex flex-wrap items-center gap-3 text-xs text-muted"><p>{study.error}. No strategy has been substituted.</p><button type="button" onClick={study.retry} className="min-h-11 text-accent underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent">Retry</button></div>}
      {study.line?.unappliedChoices ? <p role="status" className="text-xs text-muted">Saved later-street actions are preserved. Restore the required board cards to resume this line.</p> : null}
      <StrategyView node={node} board={study.target?.board ?? []} settings={settings} framed={false} compact matrixClassName="solver-workspace-matrix" pending={study.pending || Boolean(study.error)} completed={completed || Boolean(waiting)} />
      {study.node && !study.pending && <p className="text-xs text-muted">Approximate EV values · low confidence; unknown EV/equity values are shown as unavailable.</p>}
    </section>
    <aside aria-label="AI analysis panel" hidden={!analysisOpen} className="solver-ai-panel fixed z-30 overflow-hidden rounded-lg border border-border bg-surface p-4 shadow-card" onKeyDown={(e) => { if (e.key === 'Escape') setAnalysisOpen(false); }}>
      <AiPanel getSpot={getAnalysisSpot} embedded onClose={() => setAnalysisOpen(false)} />
    </aside>
  </section>;
}
