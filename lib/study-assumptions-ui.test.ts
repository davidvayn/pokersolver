import * as React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { PreflopWorkspace } from '@/app/preflop/PreflopWorkspace';
import { SolverWorkspace } from '@/components/solver/SolverWorkspace';
import { CURATED_SCENARIOS, SOLVED_SCENARIOS } from '@/data/preflop/catalog';
import { TABLE_FORMATS } from '@/lib/positions';

const noop = () => undefined;

describe('study areas keep their own rake assumptions', () => {
  it.each([
    [CURATED_SCENARIOS[0], 'Curated reference', 'Rake assumptions not verified'],
    [SOLVED_SCENARIOS[0], 'Home game', 'Rake-free push/fold charts'],
  ] as const)('shows the actual provenance for %s', (scenario, label, economics) => {
    (globalThis as typeof globalThis & { React: typeof React }).React = React;
    const html = renderToStaticMarkup(React.createElement(PreflopWorkspace, {
      format: TABLE_FORMATS.find((format) => format.seats === 2)!,
      hero: 'BTN', villain: 'BB', scenarios: [scenario], scenario,
      available: scenario.charts, active: scenario.charts[0], strategy: {}, summary: [],
      onFormat: noop, onMatchup: noop, onScenario: noop,
    }));
    expect(html).toContain(label);
    expect(html).toContain(economics);
    expect(html).toContain('Not an NL25-trained policy');
    expect(html).toContain('Online practice settings do not change these charts');
  });

  it('labels the unchanged standalone solver as rake-free and single-street', () => {
    (globalThis as typeof globalThis & { React: typeof React }).React = React;
    const html = renderToStaticMarkup(React.createElement(SolverWorkspace, {
      board: [0, 5, 10], used: new Set([0, 5, 10]), onBoardChange: noop,
      oop: {}, ip: {}, onOopChange: noop, onIpChange: noop,
      pot: 6, stack: 100, betSizes: '33, 75', raiseSizes: '100',
      onPotChange: noop, onStackChange: noop, onBetSizesChange: noop, onRaiseSizesChange: noop,
      result: null, running: false, available: true, solverError: null,
      missing: 'Add an OOP range', showSolverStats: false,
      getAnalysisSpot: () => null, onClear: noop,
    }));
    expect(html).toContain('Home game');
    expect(html).toContain('Rake-free, single-street all-in-equity model');
    expect(html).toContain('Online practice settings do not change this solver');
  });
});
