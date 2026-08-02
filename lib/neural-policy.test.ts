import { describe, expect, it, vi } from 'vitest';
import { createHand, seededRandom } from '@/lib/practice-engine';
import {
  decodeNeuralArtifact,
  encodeNeuralState,
  encodeNeuralArtifact,
  inferNeuralPolicy,
  NEURAL_ACTION_FEATURE_COUNT,
  NEURAL_ACTION_FEATURE_SCHEMA,
  NEURAL_STATE_FEATURE_COUNT,
  NEURAL_STATE_FEATURE_SCHEMA,
  NeuralArtifactClient,
  neuralLegalActions,
  type DenseNetworkDescriptor,
  type NeuralPolicyArtifact,
} from '@/lib/neural-policy';
import { buildOpponentModel } from '@/lib/opponent-model';
import type {
  ActionAbstraction,
  HandState,
  NeuralPolicyRuntime,
} from '@/lib/practice-types';

const abstraction: ActionAbstraction = {
  openSizesBb: [2, 2.5, 3],
  limpRaiseSizesBb: [3, 4, 5],
  threeBetSizesBb: [7.5, 9, 11],
  fourBetSizesBb: [18, 22, 26],
  deeperRaisePotFractions: [0.75, 1, 1.25],
  preflopRaiseCap: 4,
  flopBetPotFractions: [1 / 3, 0.75, 1.25],
  turnRiverBetPotFractions: [0.5, 1],
  postflopRaisePotFractions: [1],
  postflopRaiseCap: 1,
  includeAllIn: true,
};

function artifact(): NeuralPolicyArtifact {
  const parameters: number[] = [];
  const layer = (
    inputSize: number,
    outputSize: number,
    weights: (row: number, column: number) => number,
    biases: number[]
  ): DenseNetworkDescriptor => {
    const weightOffset = parameters.length;
    for (let row = 0; row < outputSize; row++) {
      for (let column = 0; column < inputSize; column++) {
        parameters.push(weights(row, column));
      }
    }
    const biasOffset = parameters.length;
    parameters.push(...biases);
    return {
      layers: [
        {
          inputSize,
          outputSize,
          activation: 'linear',
          weightOffset,
          biasOffset,
        },
      ],
    };
  };
  const stateAction = NEURAL_STATE_FEATURE_COUNT + NEURAL_ACTION_FEATURE_COUNT;
  const kindOffset = NEURAL_STATE_FEATURE_COUNT;
  const baselinePolicy = layer(
    stateAction,
    1,
    (_row, column) =>
      column === kindOffset + 2 ? 1 : column === kindOffset ? -1 : 0,
    [0]
  );
  const exploitResponse = layer(
    stateAction + 16,
    1,
    (_row, column) => (column === kindOffset + 5 ? 3 : 0),
    [0]
  );
  const baselineActionValue = layer(
    stateAction,
    2,
    (row, column) =>
      row === 0 && column === kindOffset + 2
        ? 0.25
        : row === 0 && column === kindOffset
          ? -0.1
          : 0,
    [0, -10]
  );
  return {
    metadata: {
      schemaVersion: 1,
      kind: 'deep-cfr-baseline-response',
      modelVersion: 'deep-cfr-test-v1',
      depthBb: 20,
      stateFeatureSchema: NEURAL_STATE_FEATURE_SCHEMA,
      stateFeatureCount: NEURAL_STATE_FEATURE_COUNT,
      actionFeatureSchema: NEURAL_ACTION_FEATURE_SCHEMA,
      actionFeatureCount: NEURAL_ACTION_FEATURE_COUNT,
      opponentProfileSchema: 'local-opponent-profile-v1',
      opponentProfileFeatureCount: 16,
      parameterCount: parameters.length,
      actionAbstraction: abstraction,
      adaptation: {
        minimumObservations: 50,
        fullConfidenceObservations: 250,
        maximumResponseWeight: 0.5,
      },
      valueCalibration: {
        standardErrorFloorBb: 0.001,
        highConfidenceMaximumBb: 0.02,
      },
      networks: {
        baselinePolicy,
        exploitResponse,
        baselineActionValue,
      },
    },
    parameters: Float32Array.from(parameters),
  };
}

async function digest(bytes: Uint8Array): Promise<string> {
  const value = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(value)]
    .map((part) => part.toString(16).padStart(2, '0'))
    .join('');
}

describe('frozen neural policy runtime', () => {
  it('canonicalizes globally permuted suits to identical state features', () => {
    const state = createHand({
      modelVersion: 'canonical-test',
      depthBb: 20,
      button: 'button-small-blind',
      hero: 'button-small-blind',
      random: seededRandom(19),
    });
    const permute = (card: number) =>
      Math.floor(card / 4) * 4 + ((card % 4 + 1) % 4);
    const permutePair = (cards: [number, number]): [number, number] => [
      permute(cards[0]),
      permute(cards[1]),
    ];
    const permuted: HandState = {
      ...state,
      holeCards: {
        'button-small-blind': permutePair(
          state.holeCards['button-small-blind']
        ),
        'big-blind': permutePair(state.holeCards['big-blind']),
      },
      board: state.board.map(permute),
    };
    expect(encodeNeuralState(permuted)).toEqual(encodeNeuralState(state));
  });

  it('encodes suit-invariant made-hand, draw, and board-texture concepts', () => {
    const initial = createHand({
      modelVersion: 'texture-test',
      depthBb: 20,
      button: 'button-small-blind',
      hero: 'button-small-blind',
      random: seededRandom(23),
    });
    const state: HandState = {
      ...initial,
      street: 'flop' as const,
      toAct: 'button-small-blind' as const,
      holeCards: {
        'button-small-blind': [48, 45],
        'big-blind': [1, 2],
      },
      board: [44, 40, 36],
    };
    const features = encodeNeuralState(state);
    const texture = features.slice(NEURAL_STATE_FEATURE_COUNT - 64);
    expect(texture[0]).toBe(1);
    expect(texture[2]).toBe(1);
    expect(texture[16]).toBe(1);
    expect(texture[21]).toBe(1);
    expect(texture[28]).toBe(1);
    expect(texture[32]).toBe(1);
    expect(texture[36]).toBe(1);
    expect(texture[48]).toBe(1);
    expect(texture[51]).toBe(1);
    expect(texture[54]).toBe(1);
    expect(texture[58]).toBe(1);
    expect(texture[59]).toBe(1);
  });

  it('round-trips the compact binary artifact and reproduces legal sizes', () => {
    const model = artifact();
    const decoded = decodeNeuralArtifact(encodeNeuralArtifact(model));
    expect(decoded.metadata.modelVersion).toBe('deep-cfr-test-v1');
    expect([...decoded.parameters]).toEqual([...model.parameters]);
    const state = createHand({
      modelVersion: model.metadata.modelVersion,
      depthBb: 20,
      button: 'button-small-blind',
      hero: 'button-small-blind',
      random: seededRandom(7),
    });
    expect(neuralLegalActions(state, abstraction).map((action) => action.id)).toEqual([
      'fold',
      'call',
      'raise-to-2.000',
      'raise-to-2.500',
      'raise-to-3.000',
      'all-in',
    ]);
    expect(
      neuralLegalActions({ ...state, raiseReopened: false }, abstraction).map(
        (action) => action.id
      )
    ).toEqual(['fold', 'call']);
  });

  it('uses baseline probabilities for grading and a capped mix for the opponent', async () => {
    const model = artifact();
    const state = createHand({
      modelVersion: model.metadata.modelVersion,
      depthBb: 20,
      button: 'button-small-blind',
      hero: 'button-small-blind',
      random: seededRandom(9),
    });
    const profile = {
      ...buildOpponentModel([], 'baseline'),
      observations: 300,
      stableEvidence: 300,
      confidence: 1,
      responseWeight: 0.25,
      reason: 'confidence-capped' as const,
    };
    const grading = await inferNeuralPolicy({
      artifact: model,
      state,
      profile,
      usage: 'grading',
    });
    const opponent = await inferNeuralPolicy({
      artifact: model,
      state,
      profile,
      usage: 'opponent',
    });
    expect(grading.trace).toBeNull();
    expect(opponent.trace?.responseWeight).toBe(0.25);
    expect(grading.node.actions.reduce((sum, action) => sum + action.probability, 0)).toBeCloseTo(1);
    expect(opponent.node.actions.reduce((sum, action) => sum + action.probability, 0)).toBeCloseTo(1);
    const baselineAllIn = grading.node.actions.find((action) => action.id === 'all-in')!;
    const servedAllIn = opponent.node.actions.find((action) => action.id === 'all-in')!;
    expect(servedAllIn.probability).toBeGreaterThan(baselineAllIn.probability);
    expect(grading.node.bestActionId).toBe('call');
    expect(grading.node.actions.every((action) => action.confidence === 'high')).toBe(true);
  });

  it('verifies immutable artifact hashes and never falls back after corruption', async () => {
    const bytes = encodeNeuralArtifact(artifact());
    const runtime: NeuralPolicyRuntime = {
      kind: 'neural-deep-cfr-v1',
      artifactUrl: '/models/practice/deep-cfr-test-v1/20bb.bin',
      artifactSha256: await digest(bytes),
      stateFeatureSchema: NEURAL_STATE_FEATURE_SCHEMA,
      actionFeatureSchema: NEURAL_ACTION_FEATURE_SCHEMA,
      opponentProfileSchema: 'local-opponent-profile-v1',
      actionAbstraction: abstraction,
      adaptation: artifact().metadata.adaptation,
    };
    const fetcher = vi.fn(async () => new Response(bytes)) as typeof fetch;
    const client = new NeuralArtifactClient(fetcher);
    await expect(
      client.load({ runtime, modelVersion: 'deep-cfr-test-v1', depthBb: 20 })
    ).resolves.toMatchObject({ metadata: { modelVersion: 'deep-cfr-test-v1' } });
    await expect(
      new NeuralArtifactClient(fetcher).load({
        runtime: { ...runtime, artifactSha256: '0'.repeat(64) },
        modelVersion: 'deep-cfr-test-v1',
        depthBb: 20,
      })
    ).rejects.toThrow('integrity');
  });
});
