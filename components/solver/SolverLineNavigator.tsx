'use client';

import { ArrowLeft, Check, Info, RotateCcw } from 'lucide-react';
import { useEffect, useRef, type CSSProperties } from 'react';
import { colorForActions } from '@/components/solver/SolverResults';
import type { SolverActionTree } from '@/lib/solver/client';
import { selectSolverLine, solverActionAmount, solverAmount } from '@/lib/solver/tree';

export function SolverLineNavigator({ tree, path, onPathChange, pending = false }: {
  tree: SolverActionTree;
  path: number[];
  onPathChange: (path: number[]) => void;
  pending?: boolean;
}) {
  const selection = selectSolverLine(tree, path);
  const statusRef = useRef<HTMLParagraphElement>(null);
  const activeCardRef = useRef<HTMLParagraphElement>(null);
  const lineKey = JSON.stringify(selection?.steps.map(({ player, label }) => [player, label]) ?? path);
  const previous = useRef(lineKey);
  useEffect(() => {
    const changed = previous.current !== lineKey;
    previous.current = lineKey;
    // Only user navigation moves focus; a board re-solve must not steal it.
    if (changed) (activeCardRef.current ?? statusRef.current)?.focus();
  }, [lineKey]);
  if (!selection) return (
    <div role="alert" className="text-sm text-muted">
      <p>This line is not in the solved tree.</p>
      <button type="button" onClick={() => onPathChange([])} disabled={pending}
        className="min-h-11 text-accent underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent">Return to start</button>
    </div>
  );
  const { node, steps } = selection;
  const completed = node.kind === 'terminal';
  const decisionIds = [...steps.map((step) => step.nodeId), ...(!completed ? [node.id] : [])];
  const status = !completed
    ? `${node.player?.toUpperCase()} to act${node.to_call > 0 ? ` · facing ${solverAmount(node.to_call)}bb` : ''}`
    : node.reason === 'fold' ? `${node.winner?.toUpperCase()} wins by fold` : 'Line complete · equity leaf';
  return (
    <section aria-label="Betting line" aria-busy={pending} className="solver-line-navigator relative min-w-0 border-y border-border py-2">
      <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1">
        <div className="flex items-center gap-1">
          <button type="button" disabled={pending || !path.length} onClick={() => onPathChange(path.slice(0, -1))}
            className="inline-flex min-h-11 items-center gap-1 rounded-md px-2 text-xs text-muted hover:text-fg disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent">
            <ArrowLeft className="h-3.5 w-3.5" aria-hidden="true" />Back
          </button>
          <button type="button" disabled={!path.length} aria-label="Reset betting line" onClick={() => onPathChange([])}
            className="inline-flex min-h-11 items-center gap-1 rounded-md px-2 text-xs text-muted hover:text-fg disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent">
            <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />Start
          </button>
          <details>
            <summary aria-label="Solver model details" className="flex min-h-11 cursor-pointer list-none items-center gap-1 rounded-md px-2 text-xs text-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent">
              <Info className="h-3.5 w-3.5" aria-hidden="true" />Model
            </summary>
            <div className="absolute left-0 top-14 z-30 w-[min(360px,calc(100vw-32px))] rounded-md border border-border bg-surface p-3 text-xs leading-5 text-muted shadow-card">
              Home game · rake-free, single-street all-in-equity model · up to {tree.max_raises} raises.
              Calls and check-check end at an equity leaf, not another betting street.
              Pot and stacks are before settlement; EV is net value from the original spot.
              Online practice settings do not change this solver.
            </div>
          </details>
        </div>
        <p ref={statusRef} tabIndex={-1} role="status" aria-live="polite" className="min-w-0 text-xs font-semibold focus:outline-none">{status}</p>
        <dl className="ml-auto flex flex-wrap gap-x-4 gap-y-1 text-xs tabular-nums">
          {(['Pot', 'OOP', 'IP'] as const).map((label, index) => (
            <div key={label} className="flex gap-1"><dt className="text-muted">{label}</dt>
              <dd className="font-semibold">{solverAmount(index === 0 ? node.pot : node.stacks[index - 1])}bb</dd>
            </div>
          ))}
        </dl>
      </div>
      <ol aria-label="Decision cards" className="solver-decision-rail">
        {decisionIds.map((id, stepIndex) => {
          const decision = tree.nodes[id];
          if (!decision.player || !decision.strategy) return null;
          const current = !completed && id === node.id;
          const selectedAction = !completed ? path[stepIndex] : undefined;
          const colors = colorForActions(decision.strategy.actions);
          return (
            <li key={id} data-solver-decision-card data-current={current}
              className={`solver-decision-card ${current ? 'border-accent' : 'border-border'}`}>
              <p ref={current ? activeCardRef : undefined} tabIndex={-1}
                className={`flex min-h-9 items-center justify-between gap-2 border-b border-border px-2 text-xs focus:outline-none ${current ? 'font-bold text-accent' : 'font-semibold text-muted'}`}>
                <span>{decision.player.toUpperCase()}</span>
                <span className="font-normal tabular-nums text-muted">{solverAmount(decision.stacks[decision.player === 'oop' ? 0 : 1])}bb</span>
              </p>
              <div role="group" aria-label={`Modeled actions, ${decision.player.toUpperCase()}, decision ${stepIndex + 1}`} className="flex flex-col gap-1 p-1">
                {decision.strategy.actions.map((label, actionIndex) => {
                  const child = tree.nodes[decision.children[actionIndex]];
                  if (!child) return null;
                  const amount = solverActionAmount(decision, child);
                  const pressed = selectedAction === actionIndex;
                  return (
                    <button key={actionIndex} type="button" disabled={pending} aria-pressed={pressed}
                      onClick={() => onPathChange([...path.slice(0, stepIndex), actionIndex])}
                      data-solver-action-index={actionIndex} data-solver-decision-index={stepIndex}
                      style={{ '--solver-action-color': colors[label] } as CSSProperties}
                      className="solver-decision-option focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent">
                      <span>{label}</span><span className="ml-auto text-[11px] font-normal tabular-nums text-muted">{amount}</span>
                      {pressed && <Check className="h-3 w-3 shrink-0" aria-hidden="true" />}
                    </button>
                  );
                })}
              </div>
            </li>
          );
        })}
      </ol>
      {!completed && !node.strategy?.rows.length && <p className="mt-2 text-xs text-muted">Negligible-reach line · conditional range values unavailable.</p>}
    </section>
  );
}
