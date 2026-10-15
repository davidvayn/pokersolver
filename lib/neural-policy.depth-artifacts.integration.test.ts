import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { applyAction, createHand, seededRandom, totalPotBb } from '@/lib/practice-engine';
import { buildOpponentModel } from '@/lib/opponent-model';
import { decodeNeuralArtifact, encodeNeuralArtifact, inferNeuralPolicy, neuralLegalActions } from '@/lib/neural-policy';

// Explicit local input; trained weights stay outside Git and active manifests.
const root = process.env.POKER_DEPTH_ARTIFACT_ROOT;
interface Run { depthBb: number; seed: number; artifacts: Array<{ binary: string }> }

describe.skipIf(!root)('frozen full-depth neural artifact compatibility', () => {
  it.each([[50, 50121], [50, 50122], [100, 100121], [100, 100122]])(
    '%ibb seed %i round-trips and infers finite, normalized policies across a complete hand',
    async (depth, seed) => {
      const summary = JSON.parse(readFileSync(`${root}/summary.json`, 'utf8')) as { results: Run[] };
      const run = summary.results.find((candidate) => candidate.depthBb === depth && candidate.seed === seed)!;
      const bytes = readFileSync(run.artifacts.at(-1)!.binary);
      const artifact = decodeNeuralArtifact(bytes);
      expect(Buffer.from(encodeNeuralArtifact(artifact)).equals(bytes)).toBe(true);
      expect(artifact.metadata.depthBb).toBe(depth);
      expect(artifact.metadata.actionAbstraction.openSizesBb).toEqual([2, 3.5, 5]);
      const profile = buildOpponentModel([], 'baseline');
      let hand = createHand({ modelVersion: artifact.metadata.modelVersion, depthBb: depth,
        button: 'button-small-blind', hero: 'button-small-blind', random: seededRandom(24716) });
      const streets = new Set<string>();
      for (let decision = 0; !hand.terminal && decision < 16; decision++) {
        streets.add(hand.street);
        const legal = neuralLegalActions(hand, artifact.metadata.actionAbstraction);
        expect(legal.filter((action) => action.kind === 'raise' || action.kind === 'bet').length).toBeLessThanOrEqual(3);
        const result = await inferNeuralPolicy({ artifact, state: hand, profile, usage: 'grading' });
        expect(result.node.actions.map((action) => action.id)).toEqual(legal.map((action) => action.id));
        expect(result.node.actions.reduce((sum, action) => sum + action.probability, 0)).toBeCloseTo(1, 8);
        expect(result.node.actions.every((action) => typeof action.evBb === 'number' && Number.isFinite(action.evBb) &&
          typeof action.standardErrorBb === 'number' && Number.isFinite(action.standardErrorBb) && action.standardErrorBb >= .005)).toBe(true);
        hand = applyAction(hand, legal.find((action) => action.kind === 'check' || action.kind === 'call')!);
        if (!hand.terminal) expect(totalPotBb(hand) + Object.values(hand.stacksBb).reduce((a, b) => a + b, 0)).toBeCloseTo(2 * depth);
      }
      expect(hand.terminal).toBe(true);
      expect([...streets]).toEqual(['preflop', 'flop', 'turn', 'river']);
      expect(Object.values(hand.stacksBb).reduce((a, b) => a + b, 0)).toBeCloseTo(2 * depth);
    }
  );
});
