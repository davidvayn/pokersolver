import * as React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { AnalystRail } from '@/components/practice/AnalystRail';
import { DEFAULT_PRACTICE_SETTINGS } from '@/lib/practice-types';
import type { PolicyManifest, PracticeDecisionRecord } from '@/lib/practice-types';
import { PUSH_FOLD_MANIFEST } from '@/lib/practice-models';
import fullHandManifests from '@/data/practice/full-hand-manifests.json';

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
  chosenActionProbability: 0.25,
  bestActionProbability: 0.75,
  offeredActionIds: ['check', 'bet-50'],
  grade: 'inaccuracy',
  confidence: 'high',
  lowConfidence: false,
};

function renderFeedback(
  value: PracticeDecisionRecord | null,
  tab: 'feedback' | 'settings' = 'feedback',
  manifest: PolicyManifest | null = null
): string {
  (globalThis as typeof globalThis & { React: typeof React }).React = React;
  return renderToStaticMarkup(
    React.createElement(AnalystRail, {
      tab,
      onTabChange: vi.fn(),
      feedback: value,
      recentHands: [],
      settings: DEFAULT_PRACTICE_SETTINGS,
      pendingSettings: null,
      onSettingsChange: vi.fn(),
      fullDepths: [],
      manifest,
      sessionDecisions: [],
      historyWarning: '',
      opponentModel: null,
    })
  );
}

describe('AnalystRail decision feedback', () => {
  it('omits deal-mode controls because every hand uses authentic random dealing', () => {
    const html = renderFeedback(null, 'settings');
    expect(html).toContain('Hero seat');
    expect(html).not.toContain('Deal mode');
    expect(html).not.toContain('Authentic random');
    expect(html).not.toContain('Adaptive (70/30)');
  });

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
    expect(html).toContain('Full policy mix');
    expect(html).toContain('Frozen frequency');
    expect(html).toContain('Frequency grade');
    expect(html).toContain('Inaccuracy');
    expect(html).toContain('Your policy frequency');
    expect(html).toContain('Top policy frequency');
    expect(html).toContain('Best EV action');
    expect(html).toContain('Best action by estimated EV');
    expect(html).toContain('Bet 50%');
    expect(html).toContain('Your estimated EV loss');
  });

  it('does not invent a zero sampling error when uncertainty is unavailable', () => {
    const lowConfidence = {
      ...feedback,
      policyActions: feedback.policyActions.map((action, index) =>
        index === 0
          ? { ...action, standardErrorBb: null, confidence: 'low' as const }
          : action
      ),
    };
    const html = renderFeedback(lowConfidence);
    expect(html).toContain('0.100bb EV · uncertainty unavailable');
    expect(html).not.toContain('0.100bb EV ± 0.000bb');
  });

  it('explains a low-confidence disagreement between policy and EV ordering', () => {
    const mismatch = {
      ...feedback,
      confidence: 'low' as const,
      lowConfidence: true,
      policyActions: feedback.policyActions.map((action, index) => ({
        ...action,
        probability: index === 0 ? 0.75 : 0.25,
        confidence: 'low' as const,
        standardErrorBb: null,
      })),
    };
    const html = renderFeedback(mismatch);
    expect(html).toContain(
      'frozen policy and approximate continuation-value oracle disagree'
    );
    expect(html).toContain('frequency grade as the strategy target');
  });

  it('preserves small policy frequencies and renders true zero as zero', () => {
    const smallMix = {
      ...feedback,
      policyActions: feedback.policyActions.map((action, index) => ({
        ...action,
        probability: index === 0 ? 0.004 : 0.996,
      })),
    };
    const smallMixHtml = renderFeedback(smallMix);
    expect(smallMixHtml).toContain('0.4%');
    expect(smallMixHtml).toContain('99.6%');
    expect(smallMixHtml).toContain('width:0.4%');

    const pureMix = {
      ...feedback,
      policyActions: feedback.policyActions.map((action, index) => ({
        ...action,
        probability: index === 0 ? 0 : 1,
      })),
    };
    const pureMixHtml = renderFeedback(pureMix);
    expect(pureMixHtml).toContain('width:0%');
    expect(pureMixHtml).toContain('100%');
  });

  it('renders prominent Model & assumptions disclosure with concise summary for push-fold', () => {
    const html = renderFeedback(null, 'feedback', PUSH_FOLD_MANIFEST);
    expect(html).toContain('Model &amp; assumptions');
    expect(html).toContain('Click to view model details');
    expect(html).toContain('Hide');
    expect(html).toContain('Approximate GTO');
    expect(html).toContain('hu-push-fold-v1');
    expect(html).toContain('169 preflop hand classes');
    expect(html).toContain('showdown equity');
    // Ensure the old long essay list and notes are not dumped
    expect(html).not.toContain('All eight bundled depths pass the v1');
  });

  it('renders concise 2-paragraph summary for full-hand experimental resolver', () => {
    const fullHandManifest = (fullHandManifests as PolicyManifest[])[0];
    const html = renderFeedback(null, 'feedback', fullHandManifest);
    expect(html).toContain('Model &amp; assumptions');
    expect(html).toContain('Experimental self-play');
    expect(html).toContain('server-side Rust continual resolver');
    expect(html).toContain('engine fails closed');
    expect(html).toContain('primary agreement');
    expect(html).toContain('Full-game exploitability certification is deferred');
    // Ensure the old verbose raw notes dump is eliminated
    expect(html).not.toContain('Active experimental practice model: exploitability is deferred, not passed.');
    expect(html).not.toContain('Complete per-round logs measure 10.282785437');
  });
});
