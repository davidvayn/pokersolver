import type {
  CompactPushFoldScenario,
  RawExactComboStrategy,
  RawPushFoldArtifact,
} from '@/data/preflop/artifacts/types';
import type {
  PreflopScenario,
  ScenarioProvenance,
} from '@/data/preflop/catalog';
import type { PreflopChart } from '@/data/preflop/ranges';
import { cardToStr, comboKey } from '@/lib/cards';

const EXPECTED_COMBOS = 1326;
const FREQUENCY_TOLERANCE = 1e-6;
const RANKS = '23456789TJQKA';
const EXPECTED_HAND_CLASSES = new Set(
  RANKS.split('').flatMap((high, highIndex) => [
    `${high}${high}`,
    ...RANKS.slice(0, highIndex)
      .split('')
      .flatMap((low) => [`${high}${low}s`, `${high}${low}o`]),
  ])
);

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function isFrequency(value: unknown): value is number {
  return isFiniteNumber(value) && value >= 0 && value <= 1;
}

function validateFrequencyPair(first: unknown, second: unknown): boolean {
  return (
    isFrequency(first) &&
    isFrequency(second) &&
    Math.abs(first + second - 1) <= FREQUENCY_TOLERANCE
  );
}

function validateCombo(
  value: unknown,
  seen: Set<number>
): string | null {
  if (!isRecord(value) || !Array.isArray(value.cards)) {
    return 'Invalid exact combo record';
  }
  const combo = value as unknown as RawExactComboStrategy;
  const [first, second] = combo.cards;
  if (
    !Number.isInteger(first) ||
    !Number.isInteger(second) ||
    first < 0 ||
    first > 51 ||
    second < 0 ||
    second > 51 ||
    first === second
  ) {
    return `Invalid cards for combo ${combo.combo_key}`;
  }
  if (comboKey(first, second) !== combo.combo_key) {
    return `Mismatched combo key ${combo.combo_key}`;
  }
  if (seen.has(combo.combo_key)) {
    return `Duplicate combo key ${combo.combo_key}`;
  }
  if (
    !isRecord(combo.small_blind) ||
    !isRecord(combo.big_blind_vs_shove) ||
    !validateFrequencyPair(
      combo.small_blind.fold,
      combo.small_blind.shove
    ) ||
    !validateFrequencyPair(
      combo.big_blind_vs_shove.fold,
      combo.big_blind_vs_shove.call
    )
  ) {
    return `Invalid action frequencies for combo ${combo.combo_key}`;
  }
  seen.add(combo.combo_key);
  return null;
}

export function validatePushFoldArtifact(
  value: unknown
): string[] {
  const errors: string[] = [];
  if (!isRecord(value)) return ['Invalid artifact root'];
  const artifact = value as Partial<RawPushFoldArtifact>;
  if (artifact.schema_version !== 1) {
    errors.push(`Unsupported artifact schema ${artifact.schema_version}`);
  }
  if (artifact.model !== 'heads-up-push-fold-monte-carlo-v1') {
    errors.push(`Unsupported solver model ${artifact.model}`);
  }
  if (!isRecord(artifact.config)) {
    errors.push('Missing solver config');
  } else {
    const config = artifact.config;
    if (
      !isFiniteNumber(config.small_blind_bb) ||
      !isFiniteNumber(config.big_blind_bb) ||
      config.small_blind_bb <= 0 ||
      config.big_blind_bb <= config.small_blind_bb ||
      !isFiniteNumber(config.effective_stack_bb) ||
      config.effective_stack_bb <= config.big_blind_bb ||
      !Number.isInteger(config.iterations) ||
      config.iterations <= 0 ||
      !Number.isInteger(config.equity_samples) ||
      config.equity_samples <= 0
    ) {
      errors.push('Invalid solver config');
    }
  }
  if (!isRecord(artifact.metrics)) {
    errors.push('Missing solver metrics');
  } else if (
    !isFiniteNumber(artifact.metrics.exploitability_bb) ||
    artifact.metrics.exploitability_bb < 0 ||
    artifact.metrics.exploitability_bb > 0.01
  ) {
    errors.push('Invalid exploitability metric');
  } else if (artifact.metrics.compatible_deals !== 1326 * 1225) {
    errors.push('Artifact did not evaluate every compatible ordered deal');
  }
  if (!isRecord(artifact.validation)) {
    errors.push('Missing solver validation');
  } else if (artifact.validation.status !== 'approximate') {
    errors.push(`Solver validation status is ${artifact.validation.status}`);
  } else {
    const checks = artifact.validation.checks;
    const requiredChecks = [
      'finite_metrics',
      'best_response_ordering',
      'strategy_probability_sums',
      'aces_shove_and_call_sanity',
      'exploitability_advisory',
    ];
    if (
      !Array.isArray(checks) ||
      requiredChecks.some(
        (name) =>
          !checks.some(
            (check) =>
              isRecord(check) && check.name === name && check.passed === true
          )
      ) ||
      checks.some(
        (check) =>
          !isRecord(check) ||
          (check.name !== 'exploitability_high_precision' &&
            check.passed !== true)
      )
    ) {
      errors.push('A required solver validation check failed');
    }
  }
  if (
    typeof artifact.artifact_id !== 'string' ||
    artifact.artifact_id.length === 0 ||
    typeof artifact.config_hash !== 'string' ||
    artifact.config_hash.length === 0 ||
    typeof artifact.solver_version !== 'string' ||
    artifact.solver_version.length === 0
  ) {
    errors.push('Missing artifact identity');
  }

  const combos = isRecord(artifact.strategies)
    ? artifact.strategies.exact_combos
    : undefined;
  if (!Array.isArray(combos)) {
    errors.push('Missing exact combo strategies');
    return errors;
  }
  if (combos.length !== EXPECTED_COMBOS) {
    errors.push(`Expected ${EXPECTED_COMBOS} combos, received ${combos.length}`);
  }
  const seen = new Set<number>();
  for (const combo of combos) {
    const error = validateCombo(combo, seen);
    if (error) errors.push(error);
  }
  return errors;
}

function weightedHandClassTokens(
  hands: CompactPushFoldScenario['hands'],
  frequencyIndex: 1 | 2
): string {
  return hands
    .map(([label, shove, call]) => {
      const weight = frequencyIndex === 1 ? shove : call;
      if (weight <= FREQUENCY_TOLERANCE) return null;
      return weight >= 1 - FREQUENCY_TOLERANCE
        ? label
        : `${label}:${weight.toFixed(6)}`;
    })
    .filter((token): token is string => token !== null)
    .join(',');
}

function createPushFoldScenario(
  metadata: {
    artifactId: string;
    configHash?: string;
    solverVersion: string;
    model: string;
    generatedAt: number;
    stack: number;
    exploitability: number;
  },
  shoveRange: string,
  callRange: string
): PreflopScenario {
  const assumptions = [
    'Heads-up, equal effective stacks, no ante, and no rake.',
    'The small blind may only fold or move all-in.',
    'Showdown equity is estimated with deterministic Monte Carlo sampling.',
    "Best-response metrics certify the independently sampled evaluation game, not exact Hold'em equity.",
  ];
  const provenance: ScenarioProvenance = {
    source: 'offline-solver',
    status: 'approximate',
    model: metadata.model,
    artifactId: metadata.artifactId,
    configHash: metadata.configHash,
    solverVersion: metadata.solverVersion,
    generatedAt: metadata.generatedAt,
    exploitabilityBb: metadata.exploitability,
    assumptions,
  };
  const charts: PreflopChart[] = [
    {
      id: `${metadata.artifactId}-sb`,
      title: `BTN / SB - Push or fold at ${metadata.stack}bb`,
      hero: 'BTN',
      vs: 'BB',
      category: 'RFI',
      formats: [2],
      actions: [
        {
          name: 'All-in',
          color: 'rgb(var(--allin))',
          range: shoveRange,
        },
      ],
    },
    {
      id: `${metadata.artifactId}-bb`,
      title: `BB vs BTN / SB shove at ${metadata.stack}bb`,
      hero: 'BB',
      vs: 'BTN',
      category: 'vs-RFI',
      formats: [2],
      actions: [
        {
          name: 'Call',
          color: 'rgb(var(--call))',
          range: callRange,
        },
      ],
    },
  ];

  return {
    id: metadata.artifactId,
    label: 'Push/fold',
    seats: 2,
    effectiveStackBb: metadata.stack,
    openingSize: { kind: 'all-in' },
    provenance,
    charts,
  };
}

function weightedComboTokens(
  combos: RawExactComboStrategy[],
  frequency: (combo: RawExactComboStrategy) => number
): string {
  return combos
    .map((combo) => {
      const weight = frequency(combo);
      if (weight <= FREQUENCY_TOLERANCE) return null;
      const cards = `${cardToStr(combo.cards[0])}${cardToStr(combo.cards[1])}`;
      return weight >= 1 - FREQUENCY_TOLERANCE
        ? cards
        : `${cards}:${weight.toFixed(6)}`;
    })
    .filter((token): token is string => token !== null)
    .join(',');
}

export function pushFoldArtifactToScenario(
  artifact: RawPushFoldArtifact
): PreflopScenario {
  const errors = validatePushFoldArtifact(artifact);
  if (errors.length > 0) {
    throw new Error(`Rejected preflop artifact: ${errors.join('; ')}`);
  }

  const stack = artifact.config.effective_stack_bb;
  const artifactId =
    artifact.artifact_id ??
    `hu-push-fold-${stack}bb-v${artifact.solver_version}-seed-${artifact.config.seed}`;
  return createPushFoldScenario(
    {
      artifactId,
      configHash: artifact.config_hash,
      solverVersion: artifact.solver_version,
      model: artifact.model,
      generatedAt: artifact.generated_at_unix_seconds,
      stack,
      exploitability: artifact.metrics.exploitability_bb,
    },
    weightedComboTokens(
      artifact.strategies.exact_combos,
      (combo) => combo.small_blind.shove
    ),
    weightedComboTokens(
      artifact.strategies.exact_combos,
      (combo) => combo.big_blind_vs_shove.call
    )
  );
}

export function compactPushFoldToScenario(
  value: unknown
): PreflopScenario {
  if (!isRecord(value)) {
    throw new Error('Rejected compact preflop scenario: invalid root');
  }
  const summary = value as Partial<CompactPushFoldScenario>;
  const hands = summary.hands;
  const actionValues = summary.action_values;
  const identityValid =
    typeof summary.artifact_id === 'string' &&
    summary.artifact_id.length > 0 &&
    typeof summary.config_hash === 'string' &&
    summary.config_hash.length > 0 &&
    typeof summary.solver_version === 'string' &&
    summary.solver_version.length > 0 &&
    typeof summary.source_sha256 === 'string' &&
    /^[a-f0-9]{64}$/.test(summary.source_sha256) &&
    typeof summary.action_values_source_sha256 === 'string' &&
    /^[a-f0-9]{64}$/.test(summary.action_values_source_sha256);
  if (
    !identityValid ||
    summary.model !== 'heads-up-push-fold-monte-carlo-v1' ||
    !isFiniteNumber(summary.generated_at_unix_seconds) ||
    !isFiniteNumber(summary.effective_stack_bb) ||
    summary.effective_stack_bb <= 1 ||
    !isFiniteNumber(summary.iterations) ||
    !Number.isInteger(summary.iterations) ||
    summary.iterations <= 0 ||
    !isFiniteNumber(summary.equity_samples) ||
    !Number.isInteger(summary.equity_samples) ||
    summary.equity_samples <= 0 ||
    !isFiniteNumber(summary.seed) ||
    !Number.isInteger(summary.seed) ||
    !isFiniteNumber(summary.exploitability_bb) ||
    summary.exploitability_bb < 0 ||
    summary.exploitability_bb > 0.01 ||
    !isFiniteNumber(summary.action_value_standard_error_upper_bound_bb) ||
    summary.action_value_standard_error_upper_bound_bb < 0 ||
    !Array.isArray(hands) ||
    hands.length !== 169 ||
    new Set(hands.map((hand) => (Array.isArray(hand) ? hand[0] : null)))
      .size !== 169 ||
    hands.some(
      (hand) =>
        !Array.isArray(hand) ||
        hand.length !== 3 ||
        !EXPECTED_HAND_CLASSES.has(hand[0]) ||
        !isFrequency(hand[1]) ||
        !isFrequency(hand[2])
    ) ||
    !Array.isArray(actionValues) ||
    actionValues.length !== 169 ||
    new Set(
      actionValues.map((hand) => (Array.isArray(hand) ? hand[0] : null))
    ).size !== 169 ||
    actionValues.some(
      (hand) =>
        !Array.isArray(hand) ||
        hand.length !== 5 ||
        !EXPECTED_HAND_CLASSES.has(hand[0]) ||
        hand.slice(1).some((value) => !isFiniteNumber(value))
    ) ||
    actionValues.some(
      ([label]) => !hands.some(([policyLabel]) => policyLabel === label)
    )
  ) {
    throw new Error(
      `Rejected compact preflop scenario ${summary.artifact_id ?? 'unknown'}`
    );
  }
  const accepted = summary as CompactPushFoldScenario;

  return createPushFoldScenario(
    {
      artifactId: accepted.artifact_id,
      configHash: accepted.config_hash,
      solverVersion: accepted.solver_version,
      model: accepted.model,
      generatedAt: accepted.generated_at_unix_seconds,
      stack: accepted.effective_stack_bb,
      exploitability: accepted.exploitability_bb,
    },
    weightedHandClassTokens(accepted.hands, 1),
    weightedHandClassTokens(accepted.hands, 2)
  );
}
