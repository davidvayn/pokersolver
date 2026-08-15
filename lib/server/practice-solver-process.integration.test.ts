import { afterAll, describe, expect, it, vi } from 'vitest';

vi.mock('server-only', () => ({}));

import {
  PRACTICE_RESOLVER_IDENTITY,
  practiceSolverProcess,
  stopPracticeSolverProcess,
} from '@/lib/server/practice-solver-process';

const runIntegration = process.env.PRACTICE_RESOLVER_INTEGRATION === '1';

describe.skipIf(!runIntegration)('pinned Rust practice resolver integration', () => {
  afterAll(async () => {
    await stopPracticeSolverProcess();
  });

  it(
    'loads the website artifacts and returns normalized preflop actions with EVs',
    async () => {
      const stateHash = 'e'.repeat(64);
      const result = (await practiceSolverProcess().query({
        stateHash,
        modelVersion: PRACTICE_RESOLVER_IDENTITY.modelVersion,
        depthBb: 20,
        privateCards: [50, 51],
        board: [],
        street: 'preflop',
        actor: 0,
        totalPotBb: 1.5,
        stacksBb: [19.5, 19],
        streetBetsBb: [0.5, 1],
        totalCommittedBb: [0.5, 1],
        lastFullRaiseBb: 1,
        raiseReopened: true,
        actions: [],
      })) as {
        schema: string;
        stateHash: string;
        modelVersion: string;
        depthBb: number;
        networkSha256: string;
        rangePolicySha256: string;
        valueNetworkSha256: string;
        preflopActionValuesSha256: string;
        maximumProbabilitySumError: number;
        actions: Array<{
          kind: string;
          amountToBb: number | null;
          probability: number;
          evBb: number | null;
          standardErrorBb: number | null;
          confidence: string;
        }>;
      };

      expect(result).toMatchObject({
        schema: 'hu-practice-continual-resolver-query-v1',
        stateHash,
        modelVersion: PRACTICE_RESOLVER_IDENTITY.modelVersion,
        depthBb: 20,
        networkSha256: PRACTICE_RESOLVER_IDENTITY.networkSha256,
        rangePolicySha256: PRACTICE_RESOLVER_IDENTITY.rangePolicySha256,
        valueNetworkSha256: PRACTICE_RESOLVER_IDENTITY.valueNetworkSha256,
        preflopActionValuesSha256:
          PRACTICE_RESOLVER_IDENTITY.preflopActionValuesSha256,
      });
      expect(result.actions).toHaveLength(8);
      expect(result.maximumProbabilitySumError).toBeLessThanOrEqual(1e-6);
      expect(
        result.actions.reduce((sum, action) => sum + action.probability, 0)
      ).toBeCloseTo(1, 10);
      for (const action of result.actions) {
        expect(action.probability).toBeGreaterThanOrEqual(0);
        expect(action.evBb).not.toBeNull();
        expect(Number.isFinite(action.evBb)).toBe(true);
        expect(action.standardErrorBb).not.toBeNull();
        expect(Number.isFinite(action.standardErrorBb)).toBe(true);
        expect(action.standardErrorBb).toBeGreaterThanOrEqual(0);
        expect(action.confidence).toBe('low');
      }
      expect(result.actions.map((action) => action.kind)).toContain('all_in');
    },
    120_000
  );
});
