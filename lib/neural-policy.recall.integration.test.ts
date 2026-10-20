import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { applyAction, createHand, seededRandom, totalPotBb } from '@/lib/practice-engine';
import { buildOpponentModel } from '@/lib/opponent-model';
import { decodeNeuralArtifact, encodeNeuralArtifact, inferNeuralPolicy, neuralLegalActions } from '@/lib/neural-policy';

// Explicit research inputs; never loads or activates these in the website.
const root = process.env.POKER_RECALL_ARTIFACT_ROOT;
describe.skipIf(!root)('matched card-memory artifact compatibility, not qualification', () => {
  it.each([['averaged',51701],['averaged',51702],['recall',51701],['recall',51702]] as const)(
    '%s seed %i supports complete hands in both seats', async (profile,seed) => {
      const summary = JSON.parse(readFileSync(`${root}/summary.json`,'utf8'));
      expect(summary.status).toBe('completed_not_activated');
      const receipt = summary.artifacts[`${profile}-${seed}`];
      const bytes = readFileSync(receipt.path);
      expect(createHash('sha256').update(bytes).digest('hex')).toBe(receipt.sha256);
      const artifact = decodeNeuralArtifact(bytes);
      expect(Buffer.from(encodeNeuralArtifact(artifact)).equals(bytes)).toBe(true);
      expect(artifact.metadata.stateFeatureCount).toBe(profile === 'recall' ? 820 : 716);
      for (const hero of ['button-small-blind','big-blind'] as const) {
        for (const line of ['passive','raised','all-in'] as const) {
          let hand = createHand({ modelVersion:artifact.metadata.modelVersion,depthBb:summary.depth_bb,
            button:'button-small-blind',hero,random:seededRandom(2919029) });
          const streets = new Set<string>(); const aggression = new Set<string>();
          for (let decision=0; !hand.terminal && decision<48; decision++) {
            streets.add(hand.street);
            const legal = neuralLegalActions(hand,artifact.metadata.actionAbstraction);
            const result = await inferNeuralPolicy({ artifact,state:hand,
              profile:buildOpponentModel([], 'baseline'),usage:'grading' });
            expect(result.node.actions.map(a=>a.id)).toEqual(legal.map(a=>a.id));
            expect(result.node.actions.reduce((sum,a)=>sum+a.probability,0)).toBeCloseTo(1,8);
            expect(result.node.actions.every(a=>Number.isFinite(a.probability) && Number.isFinite(a.evBb))).toBe(true);
            let chosen = legal.find(a=>a.kind==='check' || a.kind==='call');
            if (line==='all-in' && decision===0) chosen = legal.find(a=>a.kind==='all-in');
            if (line==='raised' && !aggression.has(hand.street)) {
              const raise = legal.find(a=>a.kind==='raise' || a.kind==='bet');
              if (raise) { chosen=raise; aggression.add(hand.street); }
            }
            expect(chosen).toBeDefined(); hand=applyAction(hand,chosen!);
            expect(Object.values(hand.stacksBb).reduce((sum,s)=>sum+s,0)+(hand.terminal?0:totalPotBb(hand))).toBeCloseTo(2*summary.depth_bb,7);
          }
          expect(hand.terminal).toBe(true);
          if (line==='passive') expect([...streets]).toEqual(['preflop','flop','turn','river']);
        }
      }
    });
});
