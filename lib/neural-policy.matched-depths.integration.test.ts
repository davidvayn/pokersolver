import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { applyAction, createHand, seededRandom, totalPotBb } from '@/lib/practice-engine';
import { buildOpponentModel } from '@/lib/opponent-model';
import { decodeNeuralArtifact, encodeNeuralArtifact, inferNeuralPolicy, neuralLegalActions } from '@/lib/neural-policy';

// Explicit local research inputs only. Passing this compatibility test does NOT
// validate strategy quality, calibrate EVs, or add any model to serving manifests.
const prefix = process.env.POKER_MATCHED_DEPTH_ARTIFACT_PREFIX;
interface MatchedSummary {
  status: string;
  depth_bb: number;
  release_eligible: boolean;
  artifacts: Record<string, { path: string; sha256: string }>;
}

const cases = [
  [50, 'control', 51701], [50, 'control', 51702],
  [50, 'averaged', 51701], [50, 'averaged', 51702],
  [100, 'control', 51701], [100, 'control', 51702],
  [100, 'averaged', 51701], [100, 'averaged', 51702],
] as const;

describe.skipIf(!prefix)('frozen matched-depth website compatibility, not promotion', () => {
  it.each(cases)('%ibb %s seed %i decodes and supports passive, raised and all-in hands in both seats',
    async (depth, configuration, seed) => {
      const summary = JSON.parse(readFileSync(`${prefix}-${depth}bb/summary.json`, 'utf8')) as MatchedSummary;
      expect(summary.status).toBe('completed_not_activated');
      expect(summary.release_eligible).toBe(false);
      expect(summary.depth_bb).toBe(depth);
      const source = summary.artifacts[`${configuration}-${seed}`];
      const bytes = readFileSync(source.path);
      expect(createHash('sha256').update(bytes).digest('hex')).toBe(source.sha256);
      const artifact = decodeNeuralArtifact(bytes);
      expect(Buffer.from(encodeNeuralArtifact(artifact)).equals(bytes)).toBe(true);
      expect(artifact.metadata.depthBb).toBe(depth);
      expect(artifact.metadata.actionAbstraction.openSizesBb).toEqual([2, 3.5, 5]);
      const profile = buildOpponentModel([], 'baseline');

      for (const hero of ['button-small-blind', 'big-blind'] as const) {
        for (const line of ['passive', 'raised', 'all-in'] as const) {
          let hand = createHand({ modelVersion: artifact.metadata.modelVersion, depthBb: depth,
            button: 'button-small-blind', hero, random: seededRandom(2717147) });
          const streets = new Set<string>();
          const aggression = new Map<string, number>();
          for (let decision = 0; !hand.terminal && decision < 48; decision++) {
            streets.add(hand.street);
            expect(hand.modelVersion).toBe(artifact.metadata.modelVersion);
            const legal = neuralLegalActions(hand, artifact.metadata.actionAbstraction);
            expect(legal.filter((action) => action.kind === 'raise' || action.kind === 'bet').length).toBeLessThanOrEqual(3);
            const result = await inferNeuralPolicy({ artifact, state: hand, profile, usage: 'grading' });
            expect(result.node.actions.map((action) => action.id)).toEqual(legal.map((action) => action.id));
            expect(result.node.actions.reduce((sum, action) => sum + action.probability, 0)).toBeCloseTo(1, 8);
            expect(result.node.actions.every((action) => Number.isFinite(action.probability) && action.probability >= 0 &&
              typeof action.evBb === 'number' && Number.isFinite(action.evBb) &&
              typeof action.standardErrorBb === 'number' && Number.isFinite(action.standardErrorBb) && action.standardErrorBb >= .005)).toBe(true);
            let chosen = legal.find((action) => action.kind === 'check' || action.kind === 'call');
            if (line === 'all-in' && decision === 0) chosen = legal.find((action) => action.kind === 'all-in');
            if (line === 'raised' && (aggression.get(hand.street) ?? 0) < 2) {
              const raise = legal.find((action) => action.kind === 'raise' || action.kind === 'bet');
              if (raise) {
                chosen = raise;
                aggression.set(hand.street, (aggression.get(hand.street) ?? 0) + 1);
              }
            }
            expect(chosen).toBeDefined();
            hand = applyAction(hand, chosen!);
            const stacks = Object.values(hand.stacksBb).reduce((sum, stack) => sum + stack, 0);
            expect(stacks + (hand.terminal ? 0 : totalPotBb(hand))).toBeCloseTo(2 * depth, 7);
          }
          expect(hand.terminal).toBe(true);
          if (line === 'passive') expect([...streets]).toEqual(['preflop', 'flop', 'turn', 'river']);
        }
      }
    });
});
