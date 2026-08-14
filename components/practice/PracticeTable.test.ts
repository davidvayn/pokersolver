import * as React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { PracticeTable } from '@/components/practice/PracticeTable';
import { createHand, seededRandom } from '@/lib/practice-engine';
import type { PolicyNode } from '@/lib/practice-types';

describe('PracticeTable decision controls', () => {
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
          probability: 0.18,
          evBb: -0.7,
          standardErrorBb: 0.625,
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
        modelLabel: 'Approximate GTO',
        revealOpponent: false,
        selectedActionId: null,
        onAction: vi.fn(),
        onContinue: vi.fn(),
        onRetry: vi.fn(),
      })
    );

    expect(html).toContain('Fold');
    expect(html).toContain('All-in 20bb');
    expect(html).not.toContain('82%');
    expect(html).not.toContain('18%');
  });
});
