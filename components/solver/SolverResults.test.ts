import * as React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import {
  SolverNerdStats,
  StrategyView,
} from '@/components/solver/SolverResults';
import type { NodeStrategy, SolverResult } from '@/lib/solver/client';

describe('StrategyView hand inspection', () => {
  it('keeps the matrix and settings while hiding stale results during a solve', () => {
    (globalThis as typeof globalThis & { React: typeof React }).React = React;
    const node: NodeStrategy = {
      title: 'OOP — first to act',
      actions: ['Check'],
      rows: [
        {
          class: 'AA',
          combos: 6,
          actions: [{ action: 'Check', freq: 1, ev: 9.25 }],
        },
      ],
    };
    const html = renderToStaticMarkup(
      React.createElement(StrategyView, {
        node,
        pending: true,
        settings: React.createElement(
          'section',
          { 'aria-label': 'Solver settings' },
          'Board Qh 7s 2c'
        ),
      })
    );

    expect(html).toContain('Board Qh 7s 2c');
    expect(html).toContain('title="AA"');
    expect(html).toContain('title="AKs"');
    expect(html).toContain('aria-busy="true"');
    expect(html).not.toContain('9.25');
    expect(html).not.toContain('data-solver-combo-inspector');
    expect(html).not.toContain('data-solver-hand-mix');
  });

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

    expect(html).toContain('data-solver-combo-inspector');
    expect(html).toContain('Combination details for Pocket Aces');
    expect(html).toContain('aria-pressed="true"');
    expect(html).toContain('aria-pressed="false"');
    expect(html).toContain('Check 60%');
    expect(html).toContain('Bet 33% 40%');
    expect(html).toContain('Bet 75% &lt;0.1%');
    expect(html).toContain('hand-class EV 1.24bb');
  });

  it('renders ComboInspector with individual suited combinations and board blockers', async () => {
    const { ComboInspector } =
      await import('@/components/solver/ComboInspector');
    const { parseBoard } = await import('@/lib/cards');

    const html = renderToStaticMarkup(
      React.createElement(ComboInspector, {
        label: 'AKs',
        row: {
          class: 'AKs',
          combos: 3,
          actions: [
            { action: 'Check', freq: 0.3, ev: 2.1 },
            { action: 'Bet 75%', freq: 0.7, ev: 2.1 },
          ],
        },
        board: parseBoard('Ah 7s 2c'),
        colors: { Check: '#3b82f6', 'Bet 75%': '#f59e0b' },
        showMetrics: true,
      })
    );

    expect(html).toContain('AKs');
    expect(html).toContain('Ace-King Suited');
    expect(html).toContain('Suited');
    expect(html).toContain('3.0 / 4 combos');
    expect(html).toContain('1 blocked');
    expect(html).toContain('Blocked by board');
    expect(html).toContain('data-solver-combo-inspector');
  });

  it('renders ComboInspector with 12 combinations for offsuit hands', async () => {
    const { ComboInspector } =
      await import('@/components/solver/ComboInspector');
    const { parseBoard, parseCard } = await import('@/lib/cards');

    const html = renderToStaticMarkup(
      React.createElement(ComboInspector, {
        label: 'AKo',
        row: {
          class: 'AKo',
          combos: 12,
          actions: [{ action: 'Check', freq: 1.0, ev: 1.5 }],
          combos_data: [
            {
              card0: parseCard('As'),
              card1: parseCard('Kh'),
              weight: 1.0,
              actions: [
                { action: 'Check', freq: 0.2, ev: 3.1 },
                { action: 'Bet 75%', freq: 0.8, ev: 3.1 },
              ],
              ev: 3.1,
              equity: 0.72,
            },
          ],
        },
        board: parseBoard('Qd 7c 2s'),
        colors: { Check: '#3b82f6', 'Bet 75%': '#f59e0b' },
        showMetrics: true,
      })
    );

    expect(html).toContain('AKo');
    expect(html).toContain('Ace-King Offsuit');
    expect(html).toContain('Offsuit');
    expect(html).toContain('+3.10bb');
    expect(html).toContain('72%');
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
