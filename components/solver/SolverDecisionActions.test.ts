import * as React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { SolverDecisionActions } from './SolverDecisionActions';
import { replayStudyLine } from '@/lib/solver/study-engine';
import { ACTIVE_FULL_HAND_MANIFESTS } from '@/lib/practice-models';
import type { LegalAction } from '@/lib/practice-types';

(globalThis as typeof globalThis & { React: typeof React }).React = React;
const manifest = ACTIVE_FULL_HAND_MANIFESTS[0];
if (manifest.runtime?.kind !== 'rust-continual-resolver-v1')
  throw new Error('Missing shared runtime');
const decision = replayStudyLine(
  {
    modelVersion: manifest.version,
    depthBb: 20,
    actionAbstraction: manifest.runtime.actionAbstraction,
  },
  [],
  []
).decisions[0];

function render(
  actions: LegalAction[],
  selected: LegalAction | null = null,
  disabled = false
) {
  return renderToStaticMarkup(
    React.createElement(SolverDecisionActions, {
      actions,
      selected,
      disabled,
      decisionIndex: 0,
      onChoose: () => {},
    })
  );
}

describe('compact solver decisions', () => {
  it('shows at most four controls while keeping the full trained action list intact', () => {
    const snapshot = JSON.stringify(decision.actions);
    expect(decision.actions.length).toBeGreaterThan(4);
    const html = render(decision.actions);
    expect(html.match(/<button\b/g)).toHaveLength(4);
    expect(html).toContain('data-solver-action-kind="fold"');
    expect(html).toContain('data-solver-action-kind="call"');
    expect(html).toContain('data-solver-action-kind="all-in"');
    expect(html).not.toContain('role="combobox"');
    expect(html).toContain('Raise 5');
    expect(html).not.toContain('Raise 2');
    expect(JSON.stringify(decision.actions)).toBe(snapshot);
  });

  it('shows the exact selected sizing and disables every control while unavailable', () => {
    const raise = decision.actions
      .filter((action) => action.kind === 'raise')
      .at(-1)!;
    const html = render(decision.actions, raise, true);
    expect(html).toContain('Raise 5');
    expect(html).toContain('aria-pressed="true"');
    expect(html.match(/disabled=""/g)).toHaveLength(4);
  });

  it('offers only the legal passive responses when facing an all-in', () => {
    const html = render(
      decision.actions.filter(
        (action) => action.kind === 'fold' || action.kind === 'call'
      )
    );
    expect(html.match(/<button\b/g)).toHaveLength(2);
    expect(html).not.toContain('role="combobox"');
    expect(html).not.toContain('data-solver-action-kind="all-in"');
  });

  it('renders a single available sizing directly alongside check and all-in', () => {
    const html = render([
      { id: 'check', kind: 'check', label: 'Check' },
      { id: 'bet-to-2', kind: 'bet', label: 'Bet to 2bb', amountToBb: 2 },
      { id: 'all-in', kind: 'all-in', label: 'All-in 20bb', amountToBb: 20 },
    ]);
    expect(html.match(/<button\b/g)).toHaveLength(3);
    expect(html).toContain('Bet 2');
    expect(html).not.toContain('role="combobox"');
  });
});
