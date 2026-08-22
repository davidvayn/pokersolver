import * as React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { StrategyView } from '@/components/solver/SolverResults';
import type { NodeStrategy } from '@/lib/solver/client';

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
