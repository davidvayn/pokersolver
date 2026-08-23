import type { Card } from '@/lib/cards';

export type PracticeMode =
  | 'full-hand'
  | 'preflop'
  | 'postflop'
  | 'push-fold';
export type FullHandDepth = 20 | 50 | 100;
export type PushFoldDepth = 2 | 3 | 5 | 8 | 10 | 12 | 15 | 20;
export type PracticeStreet = 'preflop' | 'flop' | 'turn' | 'river';
export type Seat = 'button-small-blind' | 'big-blind';
export type HeroSeatMode = 'alternate' | 'button-small-blind' | 'big-blind';
export type DealMode = 'authentic' | 'adaptive';
export type OpponentStyle = 'baseline' | 'adaptive-exploitative';
export type DecisionGoal = 'continuous' | 25 | 50 | 100;
export type ActionKind =
  | 'fold'
  | 'check'
  | 'call'
  | 'bet'
  | 'raise'
  | 'all-in';
export type ConfidenceLevel = 'high' | 'low' | 'unavailable';
export type PracticeGrade =
  | 'perfect'
  | 'excellent'
  | 'good'
  | 'inaccuracy'
  | 'mistake'
  | 'blunder';

export interface PracticeSettings {
  mode: PracticeMode;
  depthBb: FullHandDepth;
  pushFoldDepthBb: PushFoldDepth;
  postflopStreets: Array<Exclude<PracticeStreet, 'preflop'>>;
  heroSeat: HeroSeatMode;
  dealMode: DealMode;
  opponentStyle: OpponentStyle;
  decisionGoal: DecisionGoal;
}

export interface PublicAction {
  id: string;
  actor: Seat;
  street: PracticeStreet;
  kind: ActionKind;
  label: string;
  amountBb: number;
  amountToBb?: number;
  potAfterBb: number;
}

export interface LegalAction {
  id: string;
  kind: ActionKind;
  label: string;
  amountToBb?: number;
}

export interface HandResult {
  reason: 'fold' | 'showdown' | 'preflop-complete' | 'review-complete';
  winner: Seat | 'split' | null;
  potBb: number;
  netBb: Record<Seat, number>;
  winningHand?: string;
}

export interface HandState {
  id: string;
  modelVersion: string;
  depthBb: number;
  button: Seat;
  hero: Seat;
  street: PracticeStreet;
  holeCards: Record<Seat, [Card, Card]>;
  board: Card[];
  deck: Card[];
  potBb: number;
  stacksBb: Record<Seat, number>;
  streetBetsBb: Record<Seat, number>;
  totalCommittedBb: Record<Seat, number>;
  toAct: Seat | null;
  pendingActors: Seat[];
  lastFullRaiseBb: number;
  raiseReopened: boolean;
  actionHistory: PublicAction[];
  terminal: boolean;
  result: HandResult | null;
}

export interface PolicyValidationSummary {
  status: 'accepted' | 'rejected' | 'training';
  exploitabilityGateDeferred?: boolean;
  exploitabilityEstimateBb?: number;
  exploitabilityUpper99Bb?: number;
  crossSeedFrequencyMae?: number;
  primaryActionAgreement?: number;
  maximumAggregateActionDelta?: number;
  policyCoverage?: number;
  actionEvStandardErrorCoverage?: number;
  projectedStorageBytes?: number;
  rawProbabilitySumsValid?: boolean;
  quantizedProbabilitySumsValid?: boolean;
  independentSeedCount?: number;
  trainingHoursPerSeed?: [number, number];
  notes: string[];
}

export interface PolicyActionValue {
  evBb: number | null;
  standardErrorBb: number | null;
  confidence: ConfidenceLevel;
}

export interface PolicyAction extends LegalAction, PolicyActionValue {
  probability: number;
}

export interface PolicyNode {
  stateHash: string;
  actions: PolicyAction[];
  bestActionId: string | null;
  bestActionEvBb: number | null;
  reachProbability?: number;
}

export interface ActionAbstraction {
  openSizesBb: number[];
  limpRaiseSizesBb: number[];
  threeBetSizesBb: number[];
  fourBetSizesBb: number[];
  deeperRaisePotFractions: number[];
  preflopRaiseCap: number;
  flopBetPotFractions: number[];
  turnRiverBetPotFractions: number[];
  postflopRaisePotFractions: number[];
  postflopRaiseCap: number;
  includeAllIn: boolean;
}

export interface NeuralPolicyRuntime {
  kind: 'neural-deep-cfr-v1';
  artifactUrl: string;
  artifactSha256: string;
  stateFeatureSchema: 'hu-cash-trajectory-poker-aware-v4';
  actionFeatureSchema: 'hu-cash-legal-action-v1';
  opponentProfileSchema: 'local-opponent-profile-v1';
  actionAbstraction: ActionAbstraction;
  adaptation: {
    minimumObservations: number;
    fullConfidenceObservations: number;
    maximumResponseWeight: number;
  };
}

export interface ShardedPolicyRuntime {
  kind: 'binary-policy-shards-v1';
}

export interface ContinualResolverRuntime {
  kind: 'rust-continual-resolver-v1';
  endpoint: '/api/practice/resolve';
  artifactFiles: {
    networks: string;
    rangePolicy: string;
    preflopActionValues: string;
    flopValueNetwork: string;
  };
  networkSha256: string;
  rangePolicySha256: string;
  valueNetworkSha256: string;
  preflopActionValuesSha256: string;
  stateFeatureSchema: 'hu-cash-trajectory-poker-aware-v4';
  rangeFeatureSchema: 'rank-suit-invariant-combo-policy-query-v2';
  actionFeatureSchema: 'hu-cash-legal-action-v1';
  actionAbstraction: ActionAbstraction;
  dcfr: {
    positiveRegretExponent: number;
    negativeRegretExponent: number;
    strategyExponent: number;
  };
  resolver: {
    flopIterations: number;
    flopResolvedActor: 0 | 1 | null;
    flopDeploySolvedPolicy: boolean;
    turnIterations: number;
    turnResolvedActor: 0 | 1 | null;
    riverIterations: number;
    riverResolvedActor: 0 | 1 | null;
    riverSafeResolving: boolean;
    riverSafeMaxmargin: boolean;
    riverSafeIterations: number | null;
    riverSafeResolvedActor: 0 | 1 | null;
    deterministic: true;
  };
}

export interface PolicyManifest {
  schemaVersion: number;
  version: string;
  model: string;
  label: 'Approximate GTO' | 'Experimental self-play';
  subtype: 'full-hand' | 'push-fold';
  active: boolean;
  depthsBb: number[];
  generatedAt: string;
  stateSchema: string;
  shardSchema: string;
  runtime?:
    | ShardedPolicyRuntime
    | NeuralPolicyRuntime
    | ContinualResolverRuntime;
  abstraction: {
    blindsBb: [number, number];
    anteBb: number;
    rake: string;
    actionSizing: string;
    cardAbstraction: string;
    recall: string;
  };
  validation: PolicyValidationSummary;
}

export interface OpponentModelSnapshot {
  schema: 'local-opponent-profile-v1';
  version: string;
  source: 'local-indexeddb';
  observations: number;
  stableEvidence: number;
  confidence: number;
  responseWeight: number;
  maximumResponseWeight: number;
  reason:
    | 'baseline-selected'
    | 'insufficient-evidence'
    | 'unstable-evidence'
    | 'confidence-capped';
  features: number[];
}

export interface OpponentPolicyTrace {
  stateHash: string;
  modelVersion: string;
  profileVersion: string;
  evidenceCount: number;
  confidence: number;
  responseWeight: number;
  networkRoute?: 'preflop' | 'postflop';
  componentModelVersion?: string;
  baselineActions: Array<{ id: string; probability: number }>;
  responseActions: Array<{ id: string; probability: number }>;
  servedActions: Array<{ id: string; probability: number }>;
}

export interface PostflopPracticeSample {
  stateHash: string;
  depthBb: FullHandDepth;
  street: Exclude<PracticeStreet, 'preflop'>;
  state: HandState;
  replayActions: PublicAction[];
}

export interface PracticeDecisionRecord {
  id: string;
  handId: string;
  answeredAt: number;
  responseMs: number;
  modelVersion: string;
  mode: PracticeMode;
  depthBb: number;
  street: PracticeStreet;
  position: Seat;
  handBucket: string;
  facingAction: string;
  stateHash: string;
  board: Card[];
  heroCards: [Card, Card];
  chosenAction: LegalAction;
  offeredActionIds?: string[];
  policyActions: PolicyAction[];
  chosenActionProbability?: number;
  bestActionProbability?: number;
  chosenActionEvBb: number | null;
  bestActionEvBb: number | null;
  evLossBb: number | null;
  grade: PracticeGrade;
  confidence: ConfidenceLevel;
  lowConfidence: boolean;
  opponentModel?: OpponentModelSnapshot;
}

export interface PracticeHandRecord {
  id: string;
  startedAt: number;
  completedAt: number;
  modelVersion: string;
  mode: PracticeMode;
  depthBb: number;
  button: Seat;
  hero: Seat;
  heroCards: [Card, Card];
  opponentCards: [Card, Card];
  board: Card[];
  actions: PublicAction[];
  decisions: PracticeDecisionRecord[];
  opponentModel?: OpponentModelSnapshot;
  opponentPolicyQueries?: OpponentPolicyTrace[];
  result: HandResult;
}

export interface EvBreakdown {
  key: string;
  label: string;
  decisions: number;
  graded: number;
  averageEvLossBb: number | null;
  totalEvLossBb: number;
  lowConfidencePercentage: number;
}

export interface PracticeStats {
  hands: number;
  decisions: number;
  gradedDecisions: number;
  averageEvLossBb: number | null;
  totalEvLossBb: number;
  lowConfidencePercentage: number;
  averageResponseMs: number;
  trendEvLossBb: number | null;
  byStreet: EvBreakdown[];
  byStack: EvBreakdown[];
  byPosition: EvBreakdown[];
  byAction: EvBreakdown[];
  byMode: EvBreakdown[];
  bySeverity: EvBreakdown[];
  weaknesses: EvBreakdown[];
  recentCostly: PracticeDecisionRecord[];
}

export const DEFAULT_PRACTICE_SETTINGS: PracticeSettings = {
  mode: 'full-hand',
  depthBb: 20,
  pushFoldDepthBb: 20,
  postflopStreets: ['flop', 'turn', 'river'],
  heroSeat: 'alternate',
  dealMode: 'authentic',
  opponentStyle: 'adaptive-exploitative',
  decisionGoal: 'continuous',
};

export const SEATS: readonly Seat[] = [
  'button-small-blind',
  'big-blind',
] as const;

export const STREET_ORDER: readonly PracticeStreet[] = [
  'preflop',
  'flop',
  'turn',
  'river',
] as const;
