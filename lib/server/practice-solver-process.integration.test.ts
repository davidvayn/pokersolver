import { afterAll, describe, expect, it, vi } from 'vitest';
import fullHandManifests from '@/data/practice/full-hand-manifests.json';
import {
  applyAction,
  createHand,
  engineLegalActions,
  seededRandom,
} from '@/lib/practice-engine';
import { resolverQueryPayload } from '@/lib/server/practice-resolver-request';
import type {
  HandState,
  LegalAction,
  PracticeStreet,
} from '@/lib/practice-types';

vi.mock('server-only', () => ({}));

import { POST as resolvePracticeRequest } from '@/app/api/practice/resolve/route';
import {
  PRACTICE_RESOLVER_IDENTITY,
  practiceSolverProcess,
  stopPracticeSolverProcess,
} from '@/lib/server/practice-solver-process';

const runIntegration = process.env.PRACTICE_RESOLVER_INTEGRATION === '1';
const activeResolver = fullHandManifests.some(
  (manifest) =>
    manifest.version === PRACTICE_RESOLVER_IDENTITY.modelVersion &&
    manifest.active === true &&
    manifest.validation.status === 'accepted'
);

interface ResolverAction {
  kind: string;
  amountToBb: number | null;
  probability: number;
  evBb: number | null;
  standardErrorBb: number | null;
  confidence: string;
}

interface ResolverResult {
  schema: string;
  stateHash: string;
  modelVersion: string;
  depthBb: number;
  networkSha256: string;
  rangePolicySha256: string;
  valueNetworkSha256: string;
  preflopActionValuesSha256: string;
  maximumProbabilitySumError: number;
  actions: ResolverAction[];
}

function assertNormalizedActions(
  result: ResolverResult,
  uncertainty: 'measured' | 'unavailable'
): void {
  expect(result.maximumProbabilitySumError).toBeLessThanOrEqual(1e-6);
  expect(result.actions.length).toBeGreaterThanOrEqual(2);
  expect(
    result.actions.reduce((sum, action) => sum + action.probability, 0)
  ).toBeCloseTo(1, 10);
  for (const action of result.actions) {
    expect(action.probability).toBeGreaterThanOrEqual(0);
    expect(action.evBb).not.toBeNull();
    expect(Number.isFinite(action.evBb)).toBe(true);
    if (uncertainty === 'measured') {
      expect(action.standardErrorBb).not.toBeNull();
      expect(Number.isFinite(action.standardErrorBb)).toBe(true);
      expect(action.standardErrorBb).toBeGreaterThanOrEqual(0);
    } else {
      expect(action.standardErrorBb).toBeNull();
    }
    expect(action.confidence).toBe('low');
  }
}

function play(state: HandState, kind: LegalAction['kind']): HandState {
  const matching = engineLegalActions(state).filter(
    (action) => action.kind === kind
  );
  expect(matching).toHaveLength(1);
  return applyAction(state, matching[0]);
}

async function queryState(
  state: HandState,
  stateHash: string
): Promise<ResolverResult> {
  return (await practiceSolverProcess().query(
    resolverQueryPayload(
      state,
      stateHash,
      PRACTICE_RESOLVER_IDENTITY.modelVersion,
      20
    )
  )) as ResolverResult;
}

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
      })) as ResolverResult;

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
      const manifest = fullHandManifests.find(
        (candidate) =>
          candidate.version === PRACTICE_RESOLVER_IDENTITY.modelVersion
      );
      expect(manifest?.runtime).toMatchObject({
        kind: 'rust-continual-resolver-v1',
        networkSha256: PRACTICE_RESOLVER_IDENTITY.networkSha256,
        rangePolicySha256: PRACTICE_RESOLVER_IDENTITY.rangePolicySha256,
        valueNetworkSha256: PRACTICE_RESOLVER_IDENTITY.valueNetworkSha256,
        preflopActionValuesSha256:
          PRACTICE_RESOLVER_IDENTITY.preflopActionValuesSha256,
      });
      expect(result.actions).toHaveLength(8);
      assertNormalizedActions(result, 'measured');
      expect(result.actions.map((action) => action.kind)).toContain('all_in');
    },
    120_000
  );

  it(
    'serves the frozen flop mix instead of a short-run uniform artifact',
    async () => {
      const result = (await practiceSolverProcess().query({
        stateHash:
          '5f89d2ac3b0ac2a6d970ef3ec079a8812b4530683e1ed58f243648c5977a3dff',
        modelVersion: PRACTICE_RESOLVER_IDENTITY.modelVersion,
        depthBb: 20,
        privateCards: [7, 34],
        board: [13, 5, 15],
        street: 'flop',
        actor: 1,
        totalPotBb: 2,
        stacksBb: [19, 19],
        streetBetsBb: [0, 0],
        totalCommittedBb: [1, 1],
        lastFullRaiseBb: 1,
        raiseReopened: true,
        actions: [
          {
            actor: 0,
            street: 'preflop',
            kind: 'call',
            amountToBb: null,
          },
          {
            actor: 1,
            street: 'preflop',
            kind: 'check',
            amountToBb: null,
          },
        ],
      })) as ResolverResult;

      assertNormalizedActions(result, 'unavailable');
      expect(result.actions.map((action) => action.kind)).toEqual([
        'check',
        'bet',
        'bet',
        'all_in',
      ]);
      expect(result.actions[0].probability).toBeGreaterThan(0.95);
      expect(result.actions[3].probability).toBeLessThan(0.001);
      expect(new Set(result.actions.map((action) => action.probability)).size).toBe(4);
    },
    120_000
  );

  it(
    'replays a website trajectory and returns action EVs on every postflop street',
    async () => {
      let state = createHand({
        id: 'resolver-postflop-integration',
        modelVersion: PRACTICE_RESOLVER_IDENTITY.modelVersion,
        depthBb: 20,
        button: 'button-small-blind',
        hero: 'button-small-blind',
        random: seededRandom(104),
      });

      state = play(state, 'call');
      state = play(state, 'check');
      const streetStates: Array<[PracticeStreet, HandState, string]> = [
        ['flop', state, 'f'.repeat(64)],
      ];

      state = play(state, 'check');
      state = play(state, 'check');
      streetStates.push(['turn', state, 'd'.repeat(64)]);

      state = play(state, 'check');
      state = play(state, 'check');
      streetStates.push(['river', state, 'c'.repeat(64)]);

      for (const [street, snapshot, stateHash] of streetStates) {
        expect(snapshot.street).toBe(street);
        expect(snapshot.terminal).toBe(false);
        const result = await queryState(snapshot, stateHash);
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
        assertNormalizedActions(result, 'unavailable');
        expect(result.actions.map((action) => action.kind)).toContain('check');
      }
    },
    180_000
  );

  it.skipIf(!activeResolver)(
    'serves a pinned decision through the production HTTP route after activation',
    async () => {
      const stateHash = 'b'.repeat(64);
      const state = createHand({
        id: 'resolver-route-integration',
        modelVersion: PRACTICE_RESOLVER_IDENTITY.modelVersion,
        depthBb: 20,
        button: 'button-small-blind',
        hero: 'button-small-blind',
        random: seededRandom(107),
      });
      const response = await resolvePracticeRequest(
        new Request('http://localhost/api/practice/resolve', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            modelVersion: PRACTICE_RESOLVER_IDENTITY.modelVersion,
            depthBb: 20,
            stateHash,
            state,
          }),
        })
      );
      expect(response.status).toBe(200);
      expect(response.headers.get('cache-control')).toBe('private, no-store');
      const result = (await response.json()) as ResolverResult;
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
      assertNormalizedActions(result, 'measured');
    },
    120_000
  );
});
