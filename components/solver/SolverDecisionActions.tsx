'use client';

import type { CSSProperties } from 'react';
import { Check } from 'lucide-react';
import { colorForActions } from '@/components/solver/SolverResults';
import { solverActionLabel } from '@/lib/solver/strategy-display';
import type { LegalAction } from '@/lib/practice-types';

/** Show only the largest modeled sizing alongside the legal passive actions and jam. */
export function SolverDecisionActions({
  actions,
  selected,
  disabled,
  decisionIndex,
  onChoose,
}: {
  actions: readonly LegalAction[];
  selected: LegalAction | null;
  disabled: boolean;
  decisionIndex: number;
  onChoose: (action: LegalAction) => void;
}) {
  const sizes = actions.filter(
    (action) => action.kind === 'bet' || action.kind === 'raise'
  );
  const largest = sizes.reduce<LegalAction | undefined>(
    (best, action) =>
      !best || (action.amountToBb ?? 0) > (best.amountToBb ?? 0)
        ? action
        : best,
    undefined
  );
  const shown = actions.filter(
    (action) => !sizes.includes(action) || action === largest
  );
  const colors = colorForActions(shown.map((action) => action.label));
  return (
    <div
      className="solver-decision-actions"
      style={{ gridTemplateColumns: `repeat(${shown.length}, minmax(0, 1fr))` }}
    >
      {shown.map((action) => (
        <button
          key={action.id}
          type="button"
          aria-pressed={selected?.id === action.id}
          disabled={disabled}
          data-solver-action-kind={action.kind}
          data-solver-action-index={actions.indexOf(action)}
          data-solver-decision-index={decisionIndex}
          onClick={() => onChoose(action)}
          style={
            { '--solver-action-color': colors[action.label] } as CSSProperties
          }
          className="solver-decision-option focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          <span>{solverActionLabel(action.label)}</span>
          {selected?.id === action.id && (
            <Check
              className="ml-auto h-3.5 w-3.5 shrink-0"
              aria-hidden="true"
            />
          )}
        </button>
      ))}
    </div>
  );
}
