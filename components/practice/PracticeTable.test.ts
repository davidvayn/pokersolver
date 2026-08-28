import * as React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { PracticeTable } from '@/components/practice/PracticeTable';
import { applyAction, createHand, seededRandom } from '@/lib/practice-engine';
import type { PolicyNode } from '@/lib/practice-types';

describe('PracticeTable decision controls', () => {
  it('distinguishes both player stacks from the central pot with labeled chips', () => {
    (globalThis as typeof globalThis & { React: typeof React }).React = React;
    const state = createHand({
      id: 'virtual-chip-test',
      modelVersion: 'test-v1',
      depthBb: 20,
      button: 'button-small-blind',
      hero: 'button-small-blind',
      random: seededRandom(3),
    });

    const html = renderToStaticMarkup(
      React.createElement(PracticeTable, {
        state,
        node: null,
        status: 'decision',
        mode: 'full-hand',
        revealOpponent: false,
        selectedActionId: null,
        onAction: vi.fn(),
        onContinue: vi.fn(),
        onRetry: vi.fn(),
      })
    );

    expect(html.match(/practice-money-stack-player/g)).toHaveLength(2);
    expect(html.match(/practice-money-stack-pot/g)).toHaveLength(1);
    expect(html).toContain('aria-label="Hero stack, 19.5 big blinds"');
    expect(html).toContain('aria-label="Opponent stack, 19.0 big blinds"');
    expect(html).toContain('aria-label="Pot, 1.5 big blinds"');
    expect(html.match(/practice-chip-pile-secondary/g)).toHaveLength(1);
  });

  it('does not reveal policy percentages before the user acts', () => {
    (globalThis as typeof globalThis & { React: typeof React }).React = React;
    const state = createHand({
      id: 'hidden-answer-test',
      modelVersion: 'test-v1',
      depthBb: 20,
      button: 'button-small-blind',
      hero: 'button-small-blind',
      random: seededRandom(4),
    });
    const node: PolicyNode = {
      stateHash: 'a'.repeat(64),
      bestActionId: 'fold',
      bestActionEvBb: -0.5,
      actions: [
        {
          id: 'fold',
          kind: 'fold',
          label: 'Fold',
          probability: 0.82,
          evBb: -0.5,
          standardErrorBb: 0,
          confidence: 'low',
        },
        {
          id: 'all-in',
          kind: 'all-in',
          label: 'All-in 20bb',
          amountToBb: 20,
          probability: 0.08,
          evBb: -0.7,
          standardErrorBb: 0.625,
          confidence: 'low',
        },
        {
          id: 'call',
          kind: 'call',
          label: 'Call 0.5bb',
          probability: 0.1,
          evBb: -0.6,
          standardErrorBb: 0.1,
          confidence: 'low',
        },
      ],
    };

    const html = renderToStaticMarkup(
      React.createElement(PracticeTable, {
        state,
        node,
        status: 'decision',
        mode: 'push-fold',
        revealOpponent: false,
        selectedActionId: null,
        onAction: vi.fn(),
        onContinue: vi.fn(),
        onRetry: vi.fn(),
      })
    );

    expect(html).toContain('Fold');
    expect(html).toContain('Call 0.5bb');
    expect(html).not.toContain('All-in 20bb');
    expect(html).not.toContain('82%');
    expect(html).not.toContain('10%');
  });

  it('shows an explicit in-progress state without the model badge or Retry', () => {
    (globalThis as typeof globalThis & { React: typeof React }).React = React;
    const state = createHand({
      id: 'solving-copy-test',
      modelVersion: 'hu-20bb-v102-consensus-continual-resolver-experimental',
      depthBb: 20,
      button: 'button-small-blind',
      hero: 'button-small-blind',
      random: seededRandom(19),
    });

    const html = renderToStaticMarkup(
      React.createElement(PracticeTable, {
        state,
        node: null,
        status: 'solving',
        mode: 'full-hand',
        revealOpponent: false,
        selectedActionId: null,
        onAction: vi.fn(),
        onContinue: vi.fn(),
        onRetry: vi.fn(),
      })
    );

    expect(html).toContain('Currently solving');
    expect(html).toContain('Almost done');
    expect(html).not.toContain('Experimental self-play');
    expect(html).not.toContain('hu-20bb-v102');
    expect(html).not.toContain('Retry');
  });

  it('lays down the flop without showing the solving overlay during the preflop handoff', () => {
    (globalThis as typeof globalThis & { React: typeof React }).React = React;
    const initial = createHand({
      id: 'flop-handoff-test',
      modelVersion: 'test-v1',
      depthBb: 20,
      button: 'button-small-blind',
      hero: 'button-small-blind',
      random: seededRandom(23),
    });
    const called = applyAction(initial, {
      id: 'call',
      kind: 'call',
      label: 'Call 0.5bb',
      amountBb: 0.5,
    });
    const flop = applyAction(called, {
      id: 'check',
      kind: 'check',
      label: 'Check',
      amountBb: 0,
    });

    const html = renderToStaticMarkup(
      React.createElement(PracticeTable, {
        state: flop,
        node: null,
        status: 'transitioning',
        mode: 'full-hand',
        revealOpponent: false,
        selectedActionId: null,
        onAction: vi.fn(),
        onContinue: vi.fn(),
        onRetry: vi.fn(),
      })
    );

    expect(html.match(/aria-label="[2-9TJQKA] of /g)).toHaveLength(5);
    expect(html.match(/practice-board-card-dealt/g)).toHaveLength(3);
    expect(html).toContain('Flop dealt · preparing actions…');
    expect(html).not.toContain('Currently solving');
  });

  it('renders the latest table action as a prominent event and keeps it in the larger log', () => {
    (globalThis as typeof globalThis & { React: typeof React }).React = React;
    const initial = createHand({
      id: 'table-action-event-test',
      modelVersion: 'test-v1',
      depthBb: 20,
      button: 'button-small-blind',
      hero: 'button-small-blind',
      random: seededRandom(31),
    });
    const called = applyAction(initial, {
      id: 'call',
      kind: 'call',
      label: 'Call 0.5bb',
      amountToBb: 1,
    });

    const html = renderToStaticMarkup(
      React.createElement(PracticeTable, {
        state: called,
        node: null,
        status: 'feedback',
        mode: 'full-hand',
        revealOpponent: false,
        selectedActionId: 'call',
        onAction: vi.fn(),
        onContinue: vi.fn(),
        onRetry: vi.fn(),
        onOpenAnalyst: vi.fn(),
      })
    );

    expect(html).toContain('practice-action-event-call');
    expect(html).toContain('practice-action-event-hero');
    expect(html).toContain('practice-action-chip-flight');
    expect(html).toContain('Table log');
    expect(html).toContain('practice-action-log-latest');
    expect(html).toContain('Review decision');
    expect(html.match(/Call 0.5bb/g)).toHaveLength(3);
  });

  it('marks a folding seat and renders a dedicated fold event', () => {
    (globalThis as typeof globalThis & { React: typeof React }).React = React;
    const initial = createHand({
      id: 'table-fold-event-test',
      modelVersion: 'test-v1',
      depthBb: 20,
      button: 'button-small-blind',
      hero: 'big-blind',
      random: seededRandom(37),
    });
    const opened = applyAction(initial, {
      id: 'raise-2.5',
      kind: 'raise',
      label: 'Raise to 2.5bb',
      amountToBb: 2.5,
    });
    const folded = applyAction(opened, {
      id: 'fold',
      kind: 'fold',
      label: 'Fold',
    });

    const html = renderToStaticMarkup(
      React.createElement(PracticeTable, {
        state: folded,
        node: null,
        status: 'review',
        mode: 'full-hand',
        revealOpponent: true,
        selectedActionId: 'fold',
        onAction: vi.fn(),
        onContinue: vi.fn(),
        onRetry: vi.fn(),
      })
    );

    expect(html).toContain('practice-action-event-fold');
    expect(html).toContain('practice-action-event-hero');
    expect(html).toContain('practice-seat-folding');
    expect(html).toContain('You Fold. Hand review complete. Continue when ready.');
  });
});
