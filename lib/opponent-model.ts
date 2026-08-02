import type {
  ActionKind,
  NeuralPolicyRuntime,
  OpponentModelSnapshot,
  OpponentStyle,
  PracticeDecisionRecord,
  PracticeHandRecord,
  PracticeStreet,
  Seat,
} from '@/lib/practice-types';

export const OPPONENT_PROFILE_SCHEMA = 'local-opponent-profile-v1' as const;
export const OPPONENT_PROFILE_FEATURE_COUNT = 16;
export const OPPONENT_PROFILE_WINDOW = 500;

export interface OpponentAdaptationConfig {
  minimumObservations: number;
  fullConfidenceObservations: number;
  maximumResponseWeight: number;
}

export const DEFAULT_OPPONENT_ADAPTATION: OpponentAdaptationConfig = {
  minimumObservations: 50,
  fullConfidenceObservations: 250,
  maximumResponseWeight: 0.5,
};

const ACTIONS: readonly ActionKind[] = [
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
const POSITIONS: readonly Seat[] = ['button-small-blind', 'big-blind'];

function clamp(value: number, minimum = 0, maximum = 1): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function smoothedRate(successes: number, attempts: number): number {
  return (successes + 1) / (attempts + 2);
}

function smoothedDistribution(
  records: PracticeDecisionRecord[]
): number[] {
  return ACTIONS.map(
    (kind) =>
      (records.filter((record) => record.chosenAction.kind === kind).length + 1) /
      (records.length + ACTIONS.length)
  );
}

function evidenceStability(records: PracticeDecisionRecord[]): number {
  if (records.length < 2) return 0;
  const split = Math.floor(records.length / 2);
  const recent = smoothedDistribution(records.slice(0, split));
  const previous = smoothedDistribution(records.slice(split));
  const meanAbsoluteDelta = recent.reduce(
    (sum, value, index) => sum + Math.abs(value - previous[index]),
    0
  ) / ACTIONS.length;
  return clamp(1 - meanAbsoluteDelta * 3);
}

function profileVersion(records: PracticeDecisionRecord[]): string {
  let hash = 0x811c9dc5;
  for (const record of records) {
    const value = `${record.id}:${record.chosenAction.id}:${record.answeredAt}`;
    for (let index = 0; index < value.length; index++) {
      hash ^= value.charCodeAt(index);
      hash = Math.imul(hash, 0x01000193);
    }
  }
  const latest = records[0]?.answeredAt ?? 0;
  return `${OPPONENT_PROFILE_SCHEMA}-${records.length}-${latest.toString(36)}-${(hash >>> 0).toString(16).padStart(8, '0')}`;
}

function isAggressive(record: PracticeDecisionRecord): boolean {
  return (
    record.chosenAction.kind === 'bet' ||
    record.chosenAction.kind === 'raise' ||
    record.chosenAction.kind === 'all-in'
  );
}

function isFacingAggression(record: PracticeDecisionRecord): boolean {
  const facing = record.facingAction.toLowerCase();
  return facing !== 'first in' && facing !== 'checked to' && facing !== 'check';
}

function validateConfig(
  config: OpponentAdaptationConfig
): OpponentAdaptationConfig {
  if (
    !Number.isInteger(config.minimumObservations) ||
    config.minimumObservations < 1 ||
    !Number.isInteger(config.fullConfidenceObservations) ||
    config.fullConfidenceObservations <= config.minimumObservations ||
    !Number.isFinite(config.maximumResponseWeight) ||
    config.maximumResponseWeight < 0 ||
    config.maximumResponseWeight > 1
  ) {
    throw new Error('Opponent adaptation settings are invalid');
  }
  return config;
}

export function adaptationConfigForRuntime(
  runtime: NeuralPolicyRuntime | undefined
): OpponentAdaptationConfig {
  return validateConfig(runtime?.adaptation ?? DEFAULT_OPPONENT_ADAPTATION);
}

export function buildOpponentModel(
  hands: PracticeHandRecord[],
  style: OpponentStyle,
  config = DEFAULT_OPPONENT_ADAPTATION
): OpponentModelSnapshot {
  const checked = validateConfig(config);
  const records = hands
    .flatMap((hand) => hand.decisions)
    .filter((record) => record.mode !== 'push-fold')
    .sort((first, second) => second.answeredAt - first.answeredAt)
    .slice(0, OPPONENT_PROFILE_WINDOW);
  const observations = records.length;
  const stability = evidenceStability(records);
  const stableEvidence = Math.round(observations * stability);
  const actionDistribution = smoothedDistribution(records);
  const streetAggression = STREETS.map((street) => {
    const matching = records.filter((record) => record.street === street);
    return smoothedRate(matching.filter(isAggressive).length, matching.length);
  });
  const positionAggression = POSITIONS.map((position) => {
    const matching = records.filter((record) => record.position === position);
    return smoothedRate(matching.filter(isAggressive).length, matching.length);
  });
  const facingAggression = records.filter(isFacingAggression);
  const foldFacingAggression = smoothedRate(
    facingAggression.filter((record) => record.chosenAction.kind === 'fold').length,
    facingAggression.length
  );
  const raiseFacingAggression = smoothedRate(
    facingAggression.filter(isAggressive).length,
    facingAggression.length
  );
  const sampleConfidence = clamp(
    (stableEvidence - checked.minimumObservations) /
      (checked.fullConfidenceObservations - checked.minimumObservations)
  );
  const confidence = stability < 0.6 ? 0 : sampleConfidence;

  let reason: OpponentModelSnapshot['reason'];
  let responseWeight = 0;
  if (style === 'baseline') {
    reason = 'baseline-selected';
  } else if (observations < checked.minimumObservations) {
    reason = 'insufficient-evidence';
  } else if (stability < 0.6) {
    reason = 'unstable-evidence';
  } else {
    reason = 'confidence-capped';
    responseWeight = checked.maximumResponseWeight * confidence;
  }

  const features = [
    clamp(observations / checked.fullConfidenceObservations),
    stability,
    ...actionDistribution,
    ...streetAggression,
    ...positionAggression,
    foldFacingAggression,
    raiseFacingAggression,
  ];
  if (features.length !== OPPONENT_PROFILE_FEATURE_COUNT) {
    throw new Error('Opponent feature schema changed without a version bump');
  }

  return {
    schema: OPPONENT_PROFILE_SCHEMA,
    version: profileVersion(records),
    source: 'local-indexeddb',
    observations,
    stableEvidence,
    confidence,
    responseWeight,
    maximumResponseWeight: checked.maximumResponseWeight,
    reason,
    features,
  };
}
