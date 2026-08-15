import * as React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { AnalystRail } from '@/components/practice/AnalystRail';
import { DEFAULT_PRACTICE_SETTINGS } from '@/lib/practice-types';
import type { PracticeDecisionRecord } from '@/lib/practice-types';

const feedback: PracticeDecisionRecord = {
  id: 'decision-1',
  handId: 'hand-1',
  answeredAt: 1,
  responseMs: 900,
  modelVersion: 'test-v1',
  mode: 'full-hand',
  depthBb: 20,
  street: 'flop',
  position: 'button-small-blind',
  handBucket: 'AKs',
  facingAction: 'checked to',
  stateHash: 'a'.repeat(64),
  board: [0, 1, 2],
  heroCards: [50, 46],
  chosenAction: { id: 'check', kind: 'check', label: 'Check' },
  policyActions: [
    {
      id: 'check',
      kind: 'check',
      label: 'Check',
      probability: 0.25,
      evBb: 0.1,
      standardErrorBb: 0.02,
      confidence: 'high',
    },
    {
      id: 'bet-50',
      kind: 'bet',
      label: 'Bet 50%',
      amountToBb: 2,
      probability: 0.75,
      evBb: 0.25,
      standardErrorBb: 0.01,
      confidence: 'high',
    },
  ],
  chosenActionEvBb: 0.1,
  bestActionEvBb: 0.25,
  evLossBb: 0.15,
  grade: 'mistake',
  confidence: 'high',
  lowConfidence: false,
};

function renderFeedback(value: PracticeDecisionRecord | null): string {
  (globalThis as typeof globalThis & { React: typeof React }).React = React;
  return renderToStaticMarkup(
    React.createElement(AnalystRail, {
      tab: 'feedback',
      onTabChange: vi.fn(),
      feedback: value,
      recentHands: [],
      settings: DEFAULT_PRACTICE_SETTINGS,
      pendingSettings: null,
      onSettingsChange: vi.fn(),
      fullDepths: [],
      manifest: null,
      sessionDecisions: [],
      historyWarning: '',
      opponentModel: null,
    })
  );
}

describe('AnalystRail decision feedback', () => {
  it('does not show a strategy mix before feedback is available', () => {
    const html = renderFeedback(null);
    expect(html).toContain('Choose an action to see the complete policy mix');
    expect(html).not.toContain('25%');
    expect(html).not.toContain('75%');
  });

  it('shows the policy mix and an estimated loss for every action after answering', () => {
    const html = renderFeedback(feedback);
    expect(html).toContain('25%');
    expect(html).toContain('75%');
    expect(html).toContain('Estimated loss 0.150bb');
    expect(html).toContain('Estimated loss 0.000bb');
    expect(html).toContain('Best estimated EV');
    expect(html).toContain('Estimated EV loss');
  });
});
