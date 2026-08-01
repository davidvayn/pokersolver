import { describe, expect, it } from 'vitest';
import {
  decodePolicyShard,
  encodePolicyShard,
  policyNodeFromShard,
} from '@/lib/policy-codec';
import {
  decodePostflopSampleShard,
  encodePostflopSampleShard,
} from '@/lib/postflop-sample-codec';
import { applyAction, createHand, seededRandom } from '@/lib/practice-engine';
import type { PolicyNode, PostflopPracticeSample } from '@/lib/practice-types';

const node: PolicyNode = {
  stateHash: '12'.repeat(32),
  bestActionId: 'raise-2.5',
  bestActionEvBb: 0.42,
  reachProbability: 0.125,
  actions: [
    {
      id: 'fold',
      kind: 'fold',
      label: 'Fold',
      probability: 0.25,
      evBb: 0.1,
      standardErrorBb: 0.01,
      confidence: 'high',
    },
    {
      id: 'raise-2.5',
      kind: 'raise',
      label: 'Raise 2.5bb',
      amountToBb: 2.5,
      probability: 0.75,
      evBb: 0.42,
      standardErrorBb: 0.03,
      confidence: 'low',
    },
  ],
};

describe('binary policy shards', () => {
  it('round-trips probabilities, action EVs, confidence, and sizing', () => {
    const encoded = encodePolicyShard([node]);
    expect(new TextDecoder().decode(encoded.slice(0, 4))).toBe('PLP1');
    const [decoded] = decodePolicyShard(encoded);
    expect(decoded.stateHash).toBe(node.stateHash);
    expect(decoded.bestActionId).toBe('raise-2.5');
    expect(decoded.bestActionEvBb).toBeCloseTo(0.42, 5);
    expect(decoded.reachProbability).toBeCloseTo(0.125, 5);
    expect(decoded.actions[1]).toMatchObject({
      id: 'raise-2.5',
      kind: 'raise',
      label: 'Raise 2.5bb',
      amountToBb: 2.5,
      confidence: 'low',
    });
    expect(decoded.actions[1].probability).toBeCloseTo(0.75, 4);
    expect(policyNodeFromShard([decoded], node.stateHash)).toEqual(decoded);
  });

  it('rejects malformed probability sums and truncated payloads', () => {
    expect(() =>
      encodePolicyShard([
        {
          ...node,
          actions: node.actions.map((action) => ({ ...action, probability: 0.1 })),
        },
      ])
    ).toThrow('probabilities sum');
    const encoded = encodePolicyShard([node]);
    expect(() => decodePolicyShard(encoded.slice(0, -2))).toThrow('Truncated');
  });
});

describe('binary postflop sample shards', () => {
  it('round-trips reachable state and replay history without changing its hash index', () => {
    const state = createHand({
      id: 'sample-hand',
      modelVersion: 'sample-v1',
      depthBb: 20,
      button: 'button-small-blind',
      hero: 'big-blind',
      random: seededRandom(4),
    });
    const limped = applyAction(state, {
      id: 'call',
      kind: 'call',
      label: 'Call 0.5bb',
    });
    const flop = applyAction(limped, {
      id: 'check',
      kind: 'check',
      label: 'Check',
    });
    const sample: PostflopPracticeSample = {
      stateHash: '34'.repeat(32),
      depthBb: 20,
      street: 'flop',
      state: flop,
      replayActions: flop.actionHistory,
    };
    const encoded = encodePostflopSampleShard([sample]);
    expect(new TextDecoder().decode(encoded.slice(0, 4))).toBe('PLS1');
    expect(decodePostflopSampleShard(encoded)).toEqual([sample]);
    expect(() => decodePostflopSampleShard(encoded.slice(0, -1))).toThrow(
      'Truncated'
    );
  });
});
