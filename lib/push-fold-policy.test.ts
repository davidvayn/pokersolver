import { describe, expect, it } from 'vitest';
import { seededRandom } from '@/lib/practice-engine';
import {
  gradePolicyChoice,
  MIN_PRACTICE_ACTION_FREQUENCY_GAP,
  validatePolicyNode,
} from '@/lib/practice-grading';
import { createPushFoldSpot } from '@/lib/push-fold-policy';

describe('push/fold practice policy', () => {
  it.each(['button-small-blind', 'big-blind'] as const)(
    'serves finite, low-confidence EV-loss estimates for %s',
    async (hero) => {
      const spot = await createPushFoldSpot({
        depthBb: 20,
        hero,
        handNumber: 7,
        random: seededRandom(hero === 'button-small-blind' ? 17 : 29),
      });

      expect(validatePolicyNode(spot.node)).toEqual([]);
      expect(spot.node.actions).toHaveLength(2);
      expect(
        Math.abs(
          spot.node.actions[0].probability - spot.node.actions[1].probability
        )
      ).toBeGreaterThan(MIN_PRACTICE_ACTION_FREQUENCY_GAP);
      expect(
        spot.node.actions.every(
          (action) =>
            action.evBb !== null &&
            Number.isFinite(action.evBb) &&
            action.standardErrorBb !== null &&
            action.confidence === 'low'
        )
      ).toBe(true);

      const expectedBest = spot.node.actions.reduce((best, action) =>
        (action.evBb ?? -Infinity) > (best.evBb ?? -Infinity) ? action : best
      );
      expect(spot.node.bestActionId).toBe(expectedBest.id);
      expect(spot.node.bestActionEvBb).toBe(expectedBest.evBb);

      const worst = spot.node.actions.find(
        (action) => action.id !== expectedBest.id
      );
      expect(worst).toBeDefined();
      const grade = gradePolicyChoice(spot.node, worst!.id);
      expect(grade.evLossBb).not.toBeNull();
      expect(grade.evLossBb).toBeGreaterThanOrEqual(0);
      expect(grade.lowConfidence).toBe(true);
    }
  );
});
