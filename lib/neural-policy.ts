import {
  applyAction,
  canonicalPolicyHash,
  otherSeat,
  toCallBb,
  totalPotBb,
} from '@/lib/practice-engine';
import { OPPONENT_PROFILE_FEATURE_COUNT } from '@/lib/opponent-model';
import type {
  ActionAbstraction,
  ActionKind,
  HandState,
  LegalAction,
  NeuralPolicyRuntime,
  OpponentModelSnapshot,
  OpponentPolicyTrace,
  PolicyAction,
  PolicyNode,
  PracticeStreet,
  Seat,
} from '@/lib/practice-types';

const MAGIC = [0x50, 0x4c, 0x4e, 0x50] as const; // PLNP
const HEADER_BYTES = 16;
const BINARY_SCHEMA_VERSION = 1;
const EPSILON = 1e-9;
const MAX_METADATA_BYTES = 1024 * 1024;
const MAX_PARAMETER_COUNT = 32_000_000;
const validatedArtifacts = new WeakSet<object>();
const ACTION_KINDS: readonly ActionKind[] = [
  'fold',
  'check',
  'call',
  'bet',
  'raise',
  'all-in',
];
const STREETS: readonly PracticeStreet[] = [
  'preflop',
  'flop',
  'turn',
  'river',
];
const SEATS: readonly Seat[] = ['button-small-blind', 'big-blind'];

export const NEURAL_STATE_FEATURE_SCHEMA =
  'hu-cash-trajectory-poker-aware-v4' as const;
export const NEURAL_ACTION_FEATURE_SCHEMA = 'hu-cash-legal-action-v1' as const;
export const MAX_TRAJECTORY_ACTIONS = 32;
export const NEURAL_ACTION_FEATURE_COUNT = 9;
const TRAJECTORY_ACTION_FEATURES = 15;
const STATE_PREFIX_FEATURES = 124;
const POKER_FEATURE_COUNT = 48;
const TEXTURE_FEATURE_COUNT = 64;
export const NEURAL_STATE_FEATURE_COUNT =
  STATE_PREFIX_FEATURES +
  MAX_TRAJECTORY_ACTIONS * TRAJECTORY_ACTION_FEATURES +
  POKER_FEATURE_COUNT +
  TEXTURE_FEATURE_COUNT;

export type NeuralActivation = 'linear' | 'relu' | 'tanh';

export interface DenseLayerDescriptor {
  inputSize: number;
  outputSize: number;
  activation: NeuralActivation;
  weightOffset: number;
  biasOffset: number;
}

export interface DenseNetworkDescriptor {
  layers: DenseLayerDescriptor[];
}

export interface NeuralPolicyArtifactMetadata {
  schemaVersion: 1;
  kind: 'deep-cfr-baseline-response';
  modelVersion: string;
  depthBb: number;
  stateFeatureSchema: typeof NEURAL_STATE_FEATURE_SCHEMA;
  stateFeatureCount: number;
  actionFeatureSchema: typeof NEURAL_ACTION_FEATURE_SCHEMA;
  actionFeatureCount: number;
  opponentProfileSchema: 'local-opponent-profile-v1';
  opponentProfileFeatureCount: number;
  parameterCount: number;
  actionAbstraction: ActionAbstraction;
  adaptation: NeuralPolicyRuntime['adaptation'];
  valueCalibration: {
    standardErrorFloorBb: number;
    highConfidenceMaximumBb: number;
  };
  networks: {
    baselinePolicy: DenseNetworkDescriptor;
    exploitResponse: DenseNetworkDescriptor;
    baselineActionValue: DenseNetworkDescriptor;
  };
}

export interface NeuralPolicyArtifact {
  metadata: NeuralPolicyArtifactMetadata;
  parameters: Float32Array;
}

export interface NeuralPolicyResult {
  node: PolicyNode;
  trace: OpponentPolicyTrace | null;
}

function roundMoney(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function oneHot<T>(values: readonly T[], selected: T): number[] {
  return values.map((value) => (value === selected ? 1 : 0));
}

function normalized(value: number, depthBb: number): number {
  return value / Math.max(depthBb, 1);
}

function canonicalSuitMap(privateCards: number[], board: number[]): number[] {
  const privateMasks = [0, 0, 0, 0];
  const boardMasks = [0, 0, 0, 0];
  for (const card of privateCards) {
    privateMasks[card % 4] |= 1 << Math.floor(card / 4);
  }
  for (const card of board) {
    boardMasks[card % 4] |= 1 << Math.floor(card / 4);
  }
  const ordered = [0, 1, 2, 3].sort(
    (left, right) =>
      privateMasks[right] - privateMasks[left] ||
      boardMasks[right] - boardMasks[left] ||
      left - right
  );
  const mapping = [0, 0, 0, 0];
  ordered.forEach((original, canonical) => {
    mapping[original] = canonical;
  });
  return mapping;
}

function canonicalCard(card: number, suitMap: number[]): number {
  return Math.floor(card / 4) * 4 + suitMap[card % 4];
}

function rankMaskHasStraight(mask: number): boolean {
  for (let low = 0; low <= 8; low++) {
    if (((mask >> low) & 0b11111) === 0b11111) return true;
  }
  const wheel = (1 << 12) | 0b1111;
  return (mask & wheel) === wheel;
}

function bitCount(value: number): number {
  let remaining = value;
  let count = 0;
  while (remaining) {
    remaining &= remaining - 1;
    count++;
  }
  return count;
}

function straightWindowDensity(mask: number): number {
  let best = 0;
  for (let low = 0; low <= 8; low++) {
    best = Math.max(best, bitCount((mask >> low) & 0b11111));
  }
  return Math.max(best, bitCount(mask & ((1 << 12) | 0b1111)));
}

function madeHandCategory(cards: number[]): number {
  if (cards.length < 5) {
    throw new Error('Made-hand category requires at least five cards');
  }
  const rankCounts = Array<number>(13).fill(0);
  const suitMasks = Array<number>(4).fill(0);
  let rankMask = 0;
  for (const card of cards) {
    const rank = Math.floor(card / 4);
    const suit = card % 4;
    rankCounts[rank]++;
    suitMasks[suit] |= 1 << rank;
    rankMask |= 1 << rank;
  }
  if (
    suitMasks.some(
      (suitMask) => bitCount(suitMask) >= 5 && rankMaskHasStraight(suitMask)
    )
  ) {
    return 8;
  }
  if (Math.max(...rankCounts) === 4) return 7;
  const trips = rankCounts.filter((count) => count === 3).length;
  const pairs = rankCounts.filter((count) => count === 2).length;
  if (trips >= 2 || (trips >= 1 && pairs >= 1)) return 6;
  if (suitMasks.some((mask) => bitCount(mask) >= 5)) return 5;
  if (rankMaskHasStraight(rankMask)) return 4;
  if (trips) return 3;
  if (pairs >= 2) return 2;
  if (pairs === 1) return 1;
  return 0;
}

function textureFeatures(
  privateCards: number[],
  board: number[],
  street: PracticeStreet
): number[] {
  const output = Array<number>(TEXTURE_FEATURE_COUNT).fill(0);
  const holeRanks = privateCards.map((card) => Math.floor(card / 4));
  if (board.length === 0) {
    output[30] = holeRanks[0] === holeRanks[1] ? 1 : 0;
    return output;
  }

  output[0] = 1;
  output[1 + madeHandCategory([...privateCards, ...board])] = 1;
  const boardRankCounts = Array<number>(13).fill(0);
  const boardSuitCounts = Array<number>(4).fill(0);
  let boardRankMask = 0;
  for (const card of board) {
    const rank = Math.floor(card / 4);
    boardRankCounts[rank]++;
    boardSuitCounts[card % 4]++;
    boardRankMask |= 1 << rank;
  }
  const boardMaxRankCount = Math.max(...boardRankCounts);
  const boardMaxSuitCount = Math.max(...boardSuitCounts);
  const boardDensity = straightWindowDensity(boardRankMask);
  output[10 + Math.min(boardMaxRankCount - 1, 3)] = 1;
  output[14 + Math.min(boardMaxSuitCount - 1, 4)] = 1;
  output[19 + Math.min(boardDensity - 1, 4)] = 1;
  const boardRanks = board.map((card) => Math.floor(card / 4));
  const boardHigh = Math.max(...boardRanks);
  const boardLow = Math.min(...boardRanks);
  output[24 + (boardHigh >= 10 ? 2 : boardHigh >= 7 ? 1 : 0)] = 1;
  const overcards = holeRanks.filter((rank) => rank > boardHigh).length;
  output[27 + Math.min(overcards, 2)] = 1;
  const pocketPair = holeRanks[0] === holeRanks[1];
  output[30] = pocketPair ? 1 : 0;
  output[31] = pocketPair && holeRanks[0] > boardHigh ? 1 : 0;
  const matches = holeRanks.map((rank) => boardRankCounts[rank] > 0);
  output[32] = holeRanks.some((rank) => rank === boardHigh) ? 1 : 0;
  output[33] = holeRanks.some(
    (rank) =>
      rank !== boardHigh && rank !== boardLow && boardRankCounts[rank] > 0
  )
    ? 1
    : 0;
  output[34] =
    boardLow !== boardHigh && holeRanks.some((rank) => rank === boardLow) ? 1 : 0;
  output[35] = matches[0] && matches[1] ? 1 : 0;
  output[36] = matches[0] !== matches[1] ? 1 : 0;
  const boardPairs = boardRankCounts.filter((count) => count === 2).length;
  output[37] = boardPairs >= 1 ? 1 : 0;
  output[38] = boardPairs >= 2 ? 1 : 0;
  output[39] = boardRankCounts.includes(3) ? 1 : 0;
  output[40] = boardRankCounts.includes(4) ? 1 : 0;

  const fullRankCounts = [...boardRankCounts];
  const fullSuitCounts = [...boardSuitCounts];
  let fullRankMask = boardRankMask;
  for (const card of privateCards) {
    const rank = Math.floor(card / 4);
    fullRankCounts[rank]++;
    fullSuitCounts[card % 4]++;
    fullRankMask |= 1 << rank;
  }
  const fullMaxRank = Math.max(...fullRankCounts);
  const fullMaxSuit = Math.max(...fullSuitCounts);
  output[41 + Math.min(fullMaxRank - 1, 3)] = 1;
  output[45 + Math.min(fullMaxSuit - 1, 4)] = 1;
  const madeStraight = rankMaskHasStraight(fullRankMask);
  output[50] = madeStraight ? 1 : 0;
  output[51] = street !== 'river' && fullMaxSuit === 4 ? 1 : 0;
  output[52] = street === 'flop' && fullMaxSuit === 3 ? 1 : 0;
  let straightOuts = 0;
  if (street !== 'river' && !madeStraight) {
    for (let rank = 0; rank < 13; rank++) {
      if (
        (fullRankMask & (1 << rank)) === 0 &&
        rankMaskHasStraight(fullRankMask | (1 << rank))
      ) {
        straightOuts++;
      }
    }
  }
  output[53 + Math.min(straightOuts, 2)] = 1;
  output[56] = boardMaxSuitCount === 1 ? 1 : 0;
  output[57] = boardMaxSuitCount === 2 ? 1 : 0;
  output[58] = boardMaxSuitCount >= 3 ? 1 : 0;
  output[59] = boardDensity >= 3 ? 1 : 0;
  output[60] = boardDensity >= 4 ? 1 : 0;
  output[61] = boardRanks.filter((rank) => rank >= 10).length / 5;
  output[62] = boardRankCounts.filter((count) => count > 0).length / 5;
  output[63] =
    ((boardMaxRankCount >= 2 ? 1 : 0) +
      (boardMaxSuitCount >= 2 ? 1 : 0) +
      (boardDensity >= 3 ? 1 : 0)) /
    3;
  return output;
}

function aggressionCount(state: HandState): number {
  return state.actionHistory.filter(
    (action) =>
      action.street === state.street &&
      (action.kind === 'bet' ||
        action.kind === 'raise' ||
        action.kind === 'all-in')
  ).length;
}

function uniqueTargets(values: number[]): number[] {
  return [
    ...new Map(values.map((value) => [Math.round(value * 1000), roundMoney(value)])).values(),
  ];
}

function raiseTargets(
  state: HandState,
  abstraction: ActionAbstraction
): number[] {
  const actor = state.toAct as Seat;
  const aggressions = aggressionCount(state);
  if (state.street === 'preflop') {
    if (aggressions === 0 && actor === 'button-small-blind') {
      return abstraction.openSizesBb;
    }
    if (aggressions === 0) return abstraction.limpRaiseSizesBb;
    if (aggressions === 1) return abstraction.threeBetSizesBb;
    if (aggressions === 2) return abstraction.fourBetSizesBb;
    const current = Math.max(
      state.streetBetsBb['button-small-blind'],
      state.streetBetsBb['big-blind']
    );
    const potAfterCall = totalPotBb(state) + toCallBb(state, actor);
    return abstraction.deeperRaisePotFractions.map(
      (fraction) => current + potAfterCall * fraction
    );
  }

  const actorCommit = state.streetBetsBb[actor];
  if (toCallBb(state, actor) <= EPSILON) {
    const fractions =
      state.street === 'flop'
        ? abstraction.flopBetPotFractions
        : abstraction.turnRiverBetPotFractions;
    return fractions.map(
      (fraction) => actorCommit + totalPotBb(state) * fraction
    );
  }
  const opponentCommit = state.streetBetsBb[otherSeat(actor)];
  const potAfterCall = totalPotBb(state) + toCallBb(state, actor);
  return abstraction.postflopRaisePotFractions.map(
    (fraction) => opponentCommit + potAfterCall * fraction
  );
}

export function neuralLegalActions(
  state: HandState,
  abstraction: ActionAbstraction
): LegalAction[] {
  if (state.terminal || !state.toAct) return [];
  const actor = state.toAct;
  const opponent = otherSeat(actor);
  const stack = state.stacksBb[actor];
  const toCall = toCallBb(state, actor);
  const current = state.streetBetsBb[actor];
  const highest = Math.max(
    state.streetBetsBb['button-small-blind'],
    state.streetBetsBb['big-blind']
  );
  const actions: LegalAction[] =
    toCall > EPSILON
      ? [
          { id: 'fold', kind: 'fold', label: 'Fold' },
          {
            id: stack <= toCall + EPSILON ? 'call-all-in' : 'call',
            kind: 'call',
            label:
              stack <= toCall + EPSILON
                ? `Call ${stack.toFixed(1)}bb`
                : `Call ${toCall.toFixed(1)}bb`,
          },
        ]
      : [{ id: 'check', kind: 'check', label: 'Check' }];
  const cap =
    state.street === 'preflop'
      ? abstraction.preflopRaiseCap
      : abstraction.postflopRaiseCap + 1;
  if (
    stack <= toCall + EPSILON ||
    state.stacksBb[opponent] <= EPSILON ||
    state.raiseReopened === false ||
    aggressionCount(state) >= cap
  ) {
    return actions;
  }

  const maximumTo = roundMoney(current + stack);
  const minimumTo = roundMoney(highest + Math.max(state.lastFullRaiseBb, 1));
  const aggressiveKind: LegalAction['kind'] =
    highest <= EPSILON ? 'bet' : 'raise';
  for (const target of uniqueTargets(raiseTargets(state, abstraction))) {
    const capped = Math.min(target, maximumTo);
    if (capped + EPSILON < minimumTo || capped >= maximumTo - EPSILON) continue;
    const fixed = capped.toFixed(3);
    actions.push({
      id: `${aggressiveKind}-to-${fixed}`,
      kind: aggressiveKind,
      label: `${aggressiveKind === 'bet' ? 'Bet' : 'Raise'} to ${Number(fixed).toFixed(1)}bb`,
      amountToBb: Number(fixed),
    });
  }
  if (abstraction.includeAllIn) {
    actions.push({
      id: 'all-in',
      kind: 'all-in',
      label: `All-in ${maximumTo.toFixed(1)}bb`,
      amountToBb: maximumTo,
    });
  }
  for (const action of actions) applyAction(state, action);
  return actions;
}

export function encodeNeuralState(state: HandState): number[] {
  const actor = state.toAct;
  if (!actor || state.terminal) throw new Error('Cannot encode a terminal state');
  if (state.actionHistory.length > MAX_TRAJECTORY_ACTIONS) {
    throw new Error('Public trajectory exceeds the neural feature schema');
  }
  const opponent = otherSeat(actor);
  const suitMap = canonicalSuitMap(state.holeCards[actor], state.board);
  const privateCards = Array<number>(52).fill(0);
  for (const card of state.holeCards[actor]) {
    privateCards[canonicalCard(card, suitMap)] = 1;
  }
  const boardCards = Array<number>(52).fill(0);
  for (const card of state.board) {
    boardCards[canonicalCard(card, suitMap)] = 1;
  }
  const prefix = [
    ...privateCards,
    ...boardCards,
    ...oneHot(STREETS, state.street),
    ...oneHot(SEATS, actor),
    ...oneHot(SEATS, state.button),
    normalized(state.potBb, state.depthBb),
    normalized(state.stacksBb[actor], state.depthBb),
    normalized(state.stacksBb[opponent], state.depthBb),
    normalized(state.streetBetsBb[actor], state.depthBb),
    normalized(state.streetBetsBb[opponent], state.depthBb),
    normalized(state.totalCommittedBb[actor], state.depthBb),
    normalized(state.totalCommittedBb[opponent], state.depthBb),
    normalized(toCallBb(state, actor), state.depthBb),
    normalized(state.lastFullRaiseBb, state.depthBb),
    state.raiseReopened === false ? 0 : 1,
    state.board.length / 5,
    state.actionHistory.length / MAX_TRAJECTORY_ACTIONS,
  ];
  if (prefix.length !== STATE_PREFIX_FEATURES) {
    throw new Error('State feature schema changed without a version bump');
  }
  const trajectory = Array<number>(
    MAX_TRAJECTORY_ACTIONS * TRAJECTORY_ACTION_FEATURES
  ).fill(0);
  state.actionHistory.forEach((action, index) => {
    const encoded = [
      ...oneHot(SEATS, action.actor),
      ...oneHot(STREETS, action.street),
      ...oneHot(ACTION_KINDS, action.kind),
      normalized(action.amountBb, state.depthBb),
      normalized(action.amountToBb ?? 0, state.depthBb),
      normalized(action.potAfterBb, state.depthBb),
    ];
    trajectory.splice(
      index * TRAJECTORY_ACTION_FEATURES,
      TRAJECTORY_ACTION_FEATURES,
      ...encoded
    );
  });
  const poker = Array<number>(POKER_FEATURE_COUNT).fill(0);
  for (const card of state.holeCards[actor]) {
    const rank = Math.floor(card / 4);
    const suit = suitMap[card % 4];
    poker[rank] += 0.5;
    poker[26 + rank] += 0.25;
    poker[43 + suit] += 1 / 7;
  }
  for (const card of state.board) {
    const rank = Math.floor(card / 4);
    const suit = suitMap[card % 4];
    poker[13 + rank] += 0.25;
    poker[26 + rank] += 0.25;
    poker[39 + suit] += 0.2;
    poker[43 + suit] += 1 / 7;
  }
  poker[47] =
    state.holeCards[actor][0] % 4 === state.holeCards[actor][1] % 4 ? 1 : 0;
  const texture = textureFeatures(
    state.holeCards[actor],
    state.board,
    state.street
  );
  const features = [...prefix, ...trajectory, ...poker, ...texture];
  if (features.length !== NEURAL_STATE_FEATURE_COUNT) {
    throw new Error('State feature count does not match the pinned schema');
  }
  return features;
}

export function encodeNeuralAction(
  state: HandState,
  action: LegalAction
): number[] {
  const actor = state.toAct;
  if (!actor) throw new Error('Cannot encode an action without an actor');
  const current = state.streetBetsBb[actor];
  const highest = Math.max(
    state.streetBetsBb['button-small-blind'],
    state.streetBetsBb['big-blind']
  );
  const target =
    action.kind === 'call'
      ? highest
      : action.amountToBb ?? current;
  const paid = Math.max(0, target - current);
  return [
    ...oneHot(ACTION_KINDS, action.kind),
    normalized(target, state.depthBb),
    normalized(paid, state.depthBb),
    paid / Math.max(totalPotBb(state), 1),
  ];
}

function activate(value: number, activation: NeuralActivation): number {
  if (activation === 'relu') return Math.max(0, value);
  if (activation === 'tanh') return Math.tanh(value);
  return value;
}

function networkShapeErrors(
  network: DenseNetworkDescriptor,
  expectedInput: number,
  expectedOutput: number,
  parameterCount: number,
  name: string
): string[] {
  const errors: string[] = [];
  if (network.layers.length === 0) return [`${name} has no layers`];
  let input = expectedInput;
  for (const [index, layer] of network.layers.entries()) {
    if (!Number.isInteger(layer.inputSize) || layer.inputSize !== input) {
      errors.push(`${name} layer ${index} has the wrong input size`);
    }
    if (!Number.isInteger(layer.outputSize) || layer.outputSize < 1) {
      errors.push(`${name} layer ${index} has an invalid output size`);
    }
    if (!['linear', 'relu', 'tanh'].includes(layer.activation)) {
      errors.push(`${name} layer ${index} has an invalid activation`);
    }
    const weightsEnd = layer.weightOffset + layer.inputSize * layer.outputSize;
    const biasesEnd = layer.biasOffset + layer.outputSize;
    if (
      !Number.isInteger(layer.weightOffset) ||
      layer.weightOffset < 0 ||
      weightsEnd > parameterCount ||
      !Number.isInteger(layer.biasOffset) ||
      layer.biasOffset < 0 ||
      biasesEnd > parameterCount
    ) {
      errors.push(`${name} layer ${index} points outside the parameter buffer`);
    }
    input = layer.outputSize;
  }
  if (input !== expectedOutput) errors.push(`${name} has the wrong output size`);
  return errors;
}

export function validateNeuralArtifact(artifact: NeuralPolicyArtifact): string[] {
  const { metadata, parameters } = artifact;
  const errors: string[] = [];
  if (metadata.schemaVersion !== 1) errors.push('Unsupported artifact schema');
  if (metadata.kind !== 'deep-cfr-baseline-response') {
    errors.push('Unsupported neural policy kind');
  }
  if (!metadata.modelVersion || ![20, 50, 100].includes(metadata.depthBb)) {
    errors.push('Model identity or depth is invalid');
  }
  if (metadata.stateFeatureSchema !== NEURAL_STATE_FEATURE_SCHEMA) {
    errors.push('State feature schema mismatch');
  }
  if (metadata.stateFeatureCount !== NEURAL_STATE_FEATURE_COUNT) {
    errors.push('State feature count mismatch');
  }
  if (metadata.actionFeatureSchema !== NEURAL_ACTION_FEATURE_SCHEMA) {
    errors.push('Action feature schema mismatch');
  }
  if (metadata.actionFeatureCount !== NEURAL_ACTION_FEATURE_COUNT) {
    errors.push('Action feature count mismatch');
  }
  if (metadata.opponentProfileFeatureCount !== OPPONENT_PROFILE_FEATURE_COUNT) {
    errors.push('Opponent profile feature count mismatch');
  }
  if (metadata.parameterCount !== parameters.length) {
    errors.push('Parameter count mismatch');
  }
  const abstraction = metadata.actionAbstraction;
  const grids = abstraction
    ? [
        abstraction.openSizesBb,
        abstraction.limpRaiseSizesBb,
        abstraction.threeBetSizesBb,
        abstraction.fourBetSizesBb,
        abstraction.deeperRaisePotFractions,
        abstraction.flopBetPotFractions,
        abstraction.turnRiverBetPotFractions,
        abstraction.postflopRaisePotFractions,
      ]
    : [];
  if (
    grids.length !== 8 ||
    grids.some(
      (grid) =>
        !Array.isArray(grid) ||
        grid.length === 0 ||
        grid.some((value) => !Number.isFinite(value) || value <= 0) ||
        grid.some((value, index) => index > 0 && grid[index - 1] >= value)
    ) ||
    !Number.isInteger(abstraction?.preflopRaiseCap) ||
    abstraction.preflopRaiseCap < 1 ||
    !Number.isInteger(abstraction.postflopRaiseCap) ||
    abstraction.postflopRaiseCap < 1 ||
    typeof abstraction.includeAllIn !== 'boolean'
  ) {
    errors.push('Action abstraction is invalid');
  }
  const adaptation = metadata.adaptation;
  if (
    !Number.isInteger(adaptation?.minimumObservations) ||
    adaptation.minimumObservations < 1 ||
    !Number.isInteger(adaptation.fullConfidenceObservations) ||
    adaptation.fullConfidenceObservations <= adaptation.minimumObservations ||
    !Number.isFinite(adaptation.maximumResponseWeight) ||
    adaptation.maximumResponseWeight < 0 ||
    adaptation.maximumResponseWeight > 1
  ) {
    errors.push('Opponent adaptation settings are invalid');
  }
  if (
    !Number.isFinite(metadata.valueCalibration.standardErrorFloorBb) ||
    metadata.valueCalibration.standardErrorFloorBb < 0 ||
    !Number.isFinite(metadata.valueCalibration.highConfidenceMaximumBb) ||
    metadata.valueCalibration.highConfidenceMaximumBb <= 0
  ) {
    errors.push('Value calibration is invalid');
  }
  const stateAction = NEURAL_STATE_FEATURE_COUNT + NEURAL_ACTION_FEATURE_COUNT;
  errors.push(
    ...networkShapeErrors(
      metadata.networks.baselinePolicy,
      stateAction,
      1,
      parameters.length,
      'baselinePolicy'
    ),
    ...networkShapeErrors(
      metadata.networks.exploitResponse,
      stateAction + OPPONENT_PROFILE_FEATURE_COUNT,
      1,
      parameters.length,
      'exploitResponse'
    ),
    ...networkShapeErrors(
      metadata.networks.baselineActionValue,
      stateAction,
      2,
      parameters.length,
      'baselineActionValue'
    )
  );
  if (parameters.some((value) => !Number.isFinite(value))) {
    errors.push('Parameters contain a non-finite value');
  }
  return errors;
}

function assertNeuralArtifact(artifact: NeuralPolicyArtifact): void {
  if (validatedArtifacts.has(artifact)) return;
  const errors = validateNeuralArtifact(artifact);
  if (errors.length > 0) {
    throw new Error(`Neural artifact is invalid: ${errors.join('; ')}`);
  }
  validatedArtifacts.add(artifact);
}

export function runDenseNetwork(
  network: DenseNetworkDescriptor,
  parameters: Float32Array,
  input: number[]
): number[] {
  let values = input;
  for (const layer of network.layers) {
    if (values.length !== layer.inputSize) {
      throw new Error('Neural input does not match the frozen artifact');
    }
    const output = Array<number>(layer.outputSize).fill(0);
    for (let row = 0; row < layer.outputSize; row++) {
      let sum = parameters[layer.biasOffset + row];
      const offset = layer.weightOffset + row * layer.inputSize;
      for (let column = 0; column < layer.inputSize; column++) {
        sum += parameters[offset + column] * values[column];
      }
      output[row] = activate(sum, layer.activation);
    }
    values = output;
  }
  if (values.some((value) => !Number.isFinite(value))) {
    throw new Error('Neural inference produced a non-finite value');
  }
  return values;
}

function softmax(logits: number[]): number[] {
  const maximum = Math.max(...logits);
  const exponentials = logits.map((value) => Math.exp(value - maximum));
  const sum = exponentials.reduce((total, value) => total + value, 0);
  if (!Number.isFinite(sum) || sum <= 0) {
    throw new Error('Neural policy probabilities are invalid');
  }
  return exponentials.map((value) => value / sum);
}

function softplus(value: number): number {
  if (value > 20) return value;
  if (value < -20) return Math.exp(value);
  return Math.log1p(Math.exp(value));
}

export async function inferNeuralPolicy(input: {
  artifact: NeuralPolicyArtifact;
  state: HandState;
  profile: OpponentModelSnapshot;
  usage: 'grading' | 'opponent';
}): Promise<NeuralPolicyResult> {
  assertNeuralArtifact(input.artifact);
  const { metadata, parameters } = input.artifact;
  if (input.state.modelVersion !== metadata.modelVersion) {
    throw new Error('The hand and neural artifact versions do not match');
  }
  if (input.state.depthBb !== metadata.depthBb) {
    throw new Error('The hand and neural artifact depths do not match');
  }
  if (
    input.profile.schema !== metadata.opponentProfileSchema ||
    input.profile.features.length !== metadata.opponentProfileFeatureCount ||
    input.profile.features.some((feature) => !Number.isFinite(feature)) ||
    !Number.isFinite(input.profile.responseWeight) ||
    input.profile.responseWeight < 0 ||
    !Number.isFinite(input.profile.maximumResponseWeight) ||
    input.profile.maximumResponseWeight !==
      metadata.adaptation.maximumResponseWeight ||
    input.profile.responseWeight > input.profile.maximumResponseWeight ||
    !Number.isFinite(input.profile.confidence) ||
    input.profile.confidence < 0 ||
    input.profile.confidence > 1
  ) {
    throw new Error('Opponent profile schema does not match the neural artifact');
  }
  const legal = neuralLegalActions(input.state, metadata.actionAbstraction);
  if (legal.length === 0) throw new Error('Neural policy has no legal actions');
  const stateFeatures = encodeNeuralState(input.state);
  const baselineLogits: number[] = [];
  const responseLogits: number[] = [];
  const valueOutputs: number[][] = [];
  for (const action of legal) {
    const stateAction = [
      ...stateFeatures,
      ...encodeNeuralAction(input.state, action),
    ];
    baselineLogits.push(
      runDenseNetwork(metadata.networks.baselinePolicy, parameters, stateAction)[0]
    );
    valueOutputs.push(
      runDenseNetwork(
        metadata.networks.baselineActionValue,
        parameters,
        stateAction
      )
    );
    if (input.usage === 'opponent') {
      responseLogits.push(
        runDenseNetwork(metadata.networks.exploitResponse, parameters, [
          ...stateAction,
          ...input.profile.features,
        ])[0]
      );
    }
  }
  const baseline = softmax(baselineLogits);
  const response =
    input.usage === 'opponent' ? softmax(responseLogits) : baseline;
  const responseWeight =
    input.usage === 'opponent'
      ? Math.min(
          input.profile.responseWeight,
          metadata.adaptation.maximumResponseWeight
        )
      : 0;
  const served = baseline.map(
    (probability, index) =>
      probability * (1 - responseWeight) + response[index] * responseWeight
  );
  const actions: PolicyAction[] = legal.map((action, index) => {
    const [evBb, rawStandardError] = valueOutputs[index];
    const standardErrorBb =
      metadata.valueCalibration.standardErrorFloorBb + softplus(rawStandardError);
    return {
      ...action,
      probability: served[index],
      evBb,
      standardErrorBb,
      confidence:
        standardErrorBb <= metadata.valueCalibration.highConfidenceMaximumBb
          ? 'high'
          : 'low',
    };
  });
  const best = actions.reduce((current, action) =>
    current.evBb === null ||
    (action.evBb !== null && action.evBb > current.evBb)
      ? action
      : current
  );
  const stateHash = await canonicalPolicyHash(input.state);
  const trace: OpponentPolicyTrace | null =
    input.usage === 'opponent'
      ? {
          stateHash,
          modelVersion: metadata.modelVersion,
          profileVersion: input.profile.version,
          evidenceCount: input.profile.observations,
          confidence: input.profile.confidence,
          responseWeight,
          baselineActions: legal.map((action, index) => ({
            id: action.id,
            probability: baseline[index],
          })),
          responseActions: legal.map((action, index) => ({
            id: action.id,
            probability: response[index],
          })),
          servedActions: legal.map((action, index) => ({
            id: action.id,
            probability: served[index],
          })),
        }
      : null;
  return {
    node: {
      stateHash,
      actions,
      bestActionId: best.id,
      bestActionEvBb: best.evBb,
    },
    trace,
  };
}

export function encodeNeuralArtifact(
  artifact: NeuralPolicyArtifact
): Uint8Array {
  try {
    assertNeuralArtifact(artifact);
  } catch (error) {
    throw new Error(
      `Cannot encode artifact: ${error instanceof Error ? error.message : 'invalid artifact'}`
    );
  }
  const metadataBytes = new TextEncoder().encode(JSON.stringify(artifact.metadata));
  if (metadataBytes.length > MAX_METADATA_BYTES) throw new Error('Artifact metadata is too large');
  const bytes = new Uint8Array(
    HEADER_BYTES + metadataBytes.length + artifact.parameters.length * 4
  );
  bytes.set(MAGIC, 0);
  const view = new DataView(bytes.buffer);
  view.setUint16(4, BINARY_SCHEMA_VERSION, true);
  view.setUint16(6, 0, true);
  view.setUint32(8, metadataBytes.length, true);
  view.setUint32(12, artifact.parameters.length, true);
  bytes.set(metadataBytes, HEADER_BYTES);
  let offset = HEADER_BYTES + metadataBytes.length;
  for (const value of artifact.parameters) {
    view.setFloat32(offset, value, true);
    offset += 4;
  }
  return bytes;
}

export function decodeNeuralArtifact(bytes: Uint8Array): NeuralPolicyArtifact {
  if (bytes.length < HEADER_BYTES) throw new Error('Neural artifact is truncated');
  if (MAGIC.some((value, index) => bytes[index] !== value)) {
    throw new Error('Neural artifact magic is invalid');
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint16(4, true) !== BINARY_SCHEMA_VERSION) {
    throw new Error('Neural artifact binary version is unsupported');
  }
  const metadataLength = view.getUint32(8, true);
  const parameterCount = view.getUint32(12, true);
  if (metadataLength > MAX_METADATA_BYTES || parameterCount > MAX_PARAMETER_COUNT) {
    throw new Error('Neural artifact declares an unsafe size');
  }
  const expected = HEADER_BYTES + metadataLength + parameterCount * 4;
  if (bytes.length !== expected) throw new Error('Neural artifact length is invalid');
  let metadata: NeuralPolicyArtifactMetadata;
  try {
    metadata = JSON.parse(
      new TextDecoder().decode(bytes.slice(HEADER_BYTES, HEADER_BYTES + metadataLength))
    ) as NeuralPolicyArtifactMetadata;
  } catch {
    throw new Error('Neural artifact metadata is invalid JSON');
  }
  const parameters = new Float32Array(parameterCount);
  let offset = HEADER_BYTES + metadataLength;
  for (let index = 0; index < parameterCount; index++) {
    parameters[index] = view.getFloat32(offset, true);
    offset += 4;
  }
  const artifact = { metadata, parameters };
  assertNeuralArtifact(artifact);
  return artifact;
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('');
}

function sameJson(first: unknown, second: unknown): boolean {
  return JSON.stringify(first) === JSON.stringify(second);
}

export class NeuralArtifactClient {
  private cache = new Map<string, Promise<NeuralPolicyArtifact>>();

  constructor(private readonly fetcher: typeof fetch) {}

  async load(input: {
    runtime: NeuralPolicyRuntime;
    modelVersion: string;
    depthBb: number;
  }): Promise<NeuralPolicyArtifact> {
    const { runtime } = input;
    if (!/^[a-f0-9]{64}$/.test(runtime.artifactSha256)) {
      throw new Error('Neural artifact hash is invalid');
    }
    if (
      !runtime.artifactUrl.startsWith('/models/practice/') &&
      !runtime.artifactUrl.startsWith('https://')
    ) {
      throw new Error('Neural artifact URL must be immutable static content');
    }
    const key = `${runtime.artifactUrl}#${runtime.artifactSha256}`;
    let pending = this.cache.get(key);
    if (!pending) {
      pending = this.fetcher(runtime.artifactUrl, {
        cache: 'force-cache',
        credentials: 'omit',
      }).then(async (response) => {
        if (!response.ok) throw new Error('Neural model artifact is unavailable');
        const bytes = new Uint8Array(await response.arrayBuffer());
        if ((await sha256Hex(bytes)) !== runtime.artifactSha256) {
          throw new Error('Neural model artifact failed its integrity check');
        }
        const artifact = decodeNeuralArtifact(bytes);
        const metadata = artifact.metadata;
        if (
          metadata.modelVersion !== input.modelVersion ||
          metadata.depthBb !== input.depthBb ||
          metadata.stateFeatureSchema !== runtime.stateFeatureSchema ||
          metadata.actionFeatureSchema !== runtime.actionFeatureSchema ||
          metadata.opponentProfileSchema !== runtime.opponentProfileSchema ||
          !sameJson(metadata.actionAbstraction, runtime.actionAbstraction) ||
          !sameJson(metadata.adaptation, runtime.adaptation)
        ) {
          throw new Error('Neural artifact does not match the pinned manifest');
        }
        return artifact;
      });
      this.cache.set(key, pending);
    }
    try {
      return await pending;
    } catch (error) {
      this.cache.delete(key);
      throw error;
    }
  }

  retry(runtime: NeuralPolicyRuntime): void {
    this.cache.delete(`${runtime.artifactUrl}#${runtime.artifactSha256}`);
  }
}
