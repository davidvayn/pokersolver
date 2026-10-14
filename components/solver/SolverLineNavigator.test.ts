import * as React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { SolverLineNavigator } from '@/components/solver/SolverLineNavigator';
import type { SolverActionTree, SolverTreeNode } from '@/lib/solver/client';

function fixture(): SolverActionTree {
  const base: SolverTreeNode = {
    id: 0, kind: 'decision', player: 'ip', pot: 9, stacks: [97, 100],
    committed: [3, 0], to_call: 3, reach_probability: 0,
    children: [1, 2, 3], reason: null, winner: null,
    strategy: { title: 'IP', actions: ['Fold', 'Call', 'Raise 100%'], rows: [] },
  };
  const leaf: SolverTreeNode = {
    ...base, kind: 'terminal', player: null, strategy: null, children: [],
    to_call: 0, reason: 'showdown',
  };
  return { root: 0, max_raises: 3, nodes: [base,
    { ...leaf, id: 1, reason: 'fold', winner: 'oop' },
    { ...leaf, id: 2, pot: 12, committed: [3, 3], stacks: [97, 97] },
    { ...leaf, id: 3, pot: 24, committed: [3, 15], stacks: [97, 85] },
  ] };
}

function render(path: number[] = [], pending = false): string {
  (globalThis as typeof globalThis & { React: typeof React }).React = React;
  return renderToStaticMarkup(React.createElement(SolverLineNavigator, {
    tree: fixture(), path, pending, onPathChange: vi.fn(),
  }));
}

describe('SolverLineNavigator', () => {
  it('offers every modeled action even at zero reach and shows the pot, stacks and actor', () => {
    const html = render();
    expect(html.match(/data-solver-action-index=/g)).toHaveLength(3);
    expect(html).toContain('Fold');
    expect(html).toContain('Call');
    expect(html).toContain('Raise 100%');
    expect(html).toContain('to 15bb');
    expect(html).toContain('9bb');
    expect(html).toContain('97bb');
    expect(html).toContain('IP to act · facing 3bb');
    expect(html).toContain('Negligible-reach line');
    expect(html).toContain('aria-label="Modeled actions, IP, decision 1"');
    expect(html).toContain('data-solver-decision-card');
    expect(html).toContain('<details>');
    expect(html).toContain('not another betting street');
  });

  it('retains historical action cards but deselects them on a folded line', () => {
    const html = render([0]);
    expect(html).toContain('OOP wins by fold');
    expect(html.match(/data-solver-action-index=/g)).toHaveLength(3);
    expect(html).not.toContain('aria-pressed="true"');
    expect(html).not.toContain('data-current="true"');
  });

  it('labels equity leaves and prevents action clicks while results are pending', () => {
    expect(render([1])).toContain('Line complete · equity leaf');
    const html = render([], true);
    expect(html.match(/disabled=""/g)).toHaveLength(5);
  });

  it('keeps historical cards mounted during a board re-solve and allows an explicit reset', () => {
    const html = render([1], true);
    expect(html).toContain('aria-busy="true"');
    expect(html).toContain('data-solver-decision-card');
    expect(html).toContain('aria-label="Reset betting line"');
    expect(html).not.toMatch(/<button[^>]*disabled[^>]*aria-label="Reset betting line"/);
    expect(html.match(/disabled=""/g)).toHaveLength(4);
  });
});
