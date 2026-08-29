import * as React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import {
  SolverNerdStats,
  StrategyView,
} from '@/components/solver/SolverResults';
import type { NodeStrategy, SolverResult } from '@/lib/solver/client';

describe('StrategyView hand inspection', () => {
  it('renders clickable grid cells with the complete action mix in their accessible description', () => {
    (globalThis as typeof globalThis & { React: typeof React }).React = React;
    const node: NodeStrategy = {
      title: 'OOP — first to act',
      actions: ['Check', 'Bet 33%', 'Bet 75%'],
      rows: [
        {
          class: 'AA',
          combos: 1,
          actions: [
            { action: 'Check', freq: 0.6, ev: 1.24 },
            { action: 'Bet 33%', freq: 0.3996, ev: 1.24 },
            { action: 'Bet 75%', freq: 0.0004, ev: 1.24 },
          ],
        },
      ],
    };

    const html = renderToStaticMarkup(
      React.createElement(StrategyView, { node })
    );

    expect(html).toContain('Select a hand to inspect its mix');
    expect(html).toContain('aria-pressed="false"');
    expect(html).toContain('Check 60%');
    expect(html).toContain('Bet 33% 40%');
    expect(html).toContain('Bet 75% &lt;0.1%');
    expect(html).toContain('hand-class EV 1.24bb');
  });
});

describe('SolverNerdStats', () => {
  it('renders all diagnostics in its opt-in summary', () => {
    const node: NodeStrategy = {
      title: 'OOP — first to act',
      actions: ['Check'],
      rows: [],
    };
    const result: SolverResult = {
      algorithm: 'cfr_plus',
      iterations: 1000,
      exploitability_pct: 0.05,
      oop_ev: 2.72,
      ip_ev: 3.28,
      pot: 6,
      oop_combos: 100,
      ip_combos: 100,
      truncated: false,
      oop: node,
      ip: { ...node, title: 'IP — vs check' },
      exploitability_history: [],
    };

    const html = renderToStaticMarkup(
      React.createElement(SolverNerdStats, { result })
    );

    expect(html).toContain('Stats for nerds');
    expect(html).toContain('Exploitability');
    expect(html).toContain('0.05%');
    expect(html).toContain('OOP EV');
    expect(html).toContain('2.72 bb');
    expect(html).toContain('IP EV');
    expect(html).toContain('3.28 bb');
    expect(html).toContain('Model');
    expect(html).toContain('CFR+');
    expect(html).toContain('Iterations');
    expect(html).toContain('1000');
  });
});
