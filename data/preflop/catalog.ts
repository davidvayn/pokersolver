import { CHARTS, type PreflopChart } from './ranges';
import solvedScenarioData from './solved-scenarios.json';
import type { CompactPushFoldScenario } from './artifacts/types';
import { compactPushFoldToScenario } from '@/lib/preflop-artifacts';
import type { TableSeats } from '@/lib/positions';

export type OpeningSize =
  | { kind: 'raise-to'; bb: number }
  | { kind: 'all-in' }
  | { kind: 'unspecified' };

export interface ScenarioProvenance {
  source: 'curated' | 'offline-solver';
  status: 'reference' | 'validated' | 'approximate';
  model: string;
  artifactId?: string;
  configHash?: string;
  solverVersion?: string;
  generatedAt?: number;
  exploitabilityBb?: number;
  assumptions: readonly string[];
}

export interface PreflopScenario {
  id: string;
  label: string;
  seats: TableSeats;
  effectiveStackBb: number;
  openingSize: OpeningSize;
  provenance: ScenarioProvenance;
  charts: readonly PreflopChart[];
}

export interface PracticeScenarioSnapshot {
  scenarioId: string;
  label: string;
  effectiveStackBb: number;
  openingSize: OpeningSize;
  provenance: Pick<
    ScenarioProvenance,
    | 'source'
    | 'status'
    | 'model'
    | 'artifactId'
    | 'configHash'
    | 'solverVersion'
  >;
}

const CURATED_ASSUMPTIONS = [
  'Simplified cash-game reference ranges.',
  'Standardized to a 2.5bb open for consistent study; the original sizing was not recorded.',
] as const;

export const CURATED_SCENARIOS: PreflopScenario[] = ([2, 6, 9] as const).map(
  (seats) => ({
    id: `curated-${seats}-max-100bb`,
    label: 'Curated baseline',
    seats,
    effectiveStackBb: 100,
    openingSize: { kind: 'raise-to', bb: 2.5 },
    provenance: {
      source: 'curated',
      status: 'reference',
      model: 'curated-baseline',
      assumptions: CURATED_ASSUMPTIONS,
    },
    charts: CHARTS.filter((chart) => chart.formats.includes(seats)),
  })
);

export const SOLVED_SCENARIOS: PreflopScenario[] = (
  solvedScenarioData as unknown as CompactPushFoldScenario[]
).map(compactPushFoldToScenario);

export const PREFLOP_SCENARIOS: PreflopScenario[] = [
  ...CURATED_SCENARIOS,
  ...SOLVED_SCENARIOS,
];

export function scenariosForSeats(seats: TableSeats): PreflopScenario[] {
  return PREFLOP_SCENARIOS.filter((scenario) => scenario.seats === seats);
}

export function defaultScenarioForSeats(seats: TableSeats): PreflopScenario {
  const scenarios = scenariosForSeats(seats);
  const solved = scenarios.find(
    (scenario) => scenario.provenance.status === 'validated'
  );
  const scenario = solved ?? scenarios[0];
  if (!scenario) throw new Error(`No preflop scenario for ${seats} seats`);
  return scenario;
}

export function scenarioSnapshot(
  scenario: PreflopScenario
): PracticeScenarioSnapshot {
  const {
    source,
    status,
    model,
    artifactId,
    configHash,
    solverVersion,
  } = scenario.provenance;
  return {
    scenarioId: scenario.id,
    label: scenario.label,
    effectiveStackBb: scenario.effectiveStackBb,
    openingSize: scenario.openingSize,
    provenance: {
      source,
      status,
      model,
      artifactId,
      configHash,
      solverVersion,
    },
  };
}

export function openingSizeLabel(openingSize: OpeningSize): string {
  if (openingSize.kind === 'all-in') return 'All-in';
  if (openingSize.kind === 'raise-to') return `${openingSize.bb}bb open`;
  return 'Size not specified';
}
