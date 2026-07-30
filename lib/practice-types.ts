import type { PracticeScenarioSnapshot } from '@/data/preflop/catalog';
import type {
  ChartActionName,
  PreflopChart,
} from '@/data/preflop/ranges';
import type { Position, TableSeats } from '@/lib/positions';

export type PracticeAction = 'Fold' | ChartActionName;
export type PracticeCategory = PreflopChart['category'];

export interface PracticeRules {
  seats: TableSeats;
  scenarioId: string;
  categories: PracticeCategory[];
  positions: Position[];
  questionCount: number;
}

export interface ActionFrequency {
  action: PracticeAction;
  frequency: number;
}

export interface PracticeQuestion {
  id: string;
  chartId: string;
  category: PracticeCategory;
  seats: TableSeats;
  hero: Position;
  villain?: Position;
  scenario: PracticeScenarioSnapshot;
  handClass: string;
  options: PracticeAction[];
  strategy: ActionFrequency[];
  correctActions: PracticeAction[];
  recommendedAction: PracticeAction;
}

export interface PracticeRecord {
  id: string;
  answeredAt: number;
  chartId: string;
  category: PracticeCategory;
  seats: TableSeats;
  hero: Position;
  villain?: Position;
  scenario?: PracticeScenarioSnapshot;
  handClass: string;
  chosenAction: PracticeAction;
  recommendedAction: PracticeAction;
  correct: boolean;
  responseMs: number;
}

export interface StatBreakdown {
  key: string;
  label: string;
  attempts: number;
  correct: number;
  accuracy: number;
}

export interface PracticeStats {
  total: number;
  correct: number;
  accuracy: number;
  averageResponseMs: number;
  streakDays: number;
  trend: number;
  byFormat: StatBreakdown[];
  byPosition: StatBreakdown[];
  byCategory: StatBreakdown[];
  byAction: StatBreakdown[];
  byScenario: StatBreakdown[];
  weaknesses: StatBreakdown[];
  recent: PracticeRecord[];
}
