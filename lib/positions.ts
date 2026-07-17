// Table positions for 6-max and full-ring, with helpers for the position UI.

export type Position = 'UTG' | 'UTG1' | 'MP' | 'LJ' | 'HJ' | 'CO' | 'BTN' | 'SB' | 'BB';

export interface TableFormat {
  seats: number;
  positions: Position[];
}

export const SIX_MAX: TableFormat = {
  seats: 6,
  positions: ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB'],
};

export const FULL_RING: TableFormat = {
  seats: 9,
  positions: ['UTG', 'UTG1', 'MP', 'LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB'],
};

export const POSITION_LABELS: Record<Position, string> = {
  UTG: 'UTG',
  UTG1: 'UTG+1',
  MP: 'MP',
  LJ: 'LJ',
  HJ: 'HJ',
  CO: 'CO',
  BTN: 'BTN',
  SB: 'SB',
  BB: 'BB',
};

export const POSITION_FULL: Record<Position, string> = {
  UTG: 'Under the Gun',
  UTG1: 'Under the Gun +1',
  MP: 'Middle Position',
  LJ: 'Lojack',
  HJ: 'Hijack',
  CO: 'Cutoff',
  BTN: 'Button',
  SB: 'Small Blind',
  BB: 'Big Blind',
};
