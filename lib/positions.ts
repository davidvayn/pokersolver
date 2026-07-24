// Table positions for heads-up, 6-max, and full-ring, with display helpers.

export type Position = 'UTG' | 'UTG1' | 'MP' | 'LJ' | 'HJ' | 'CO' | 'BTN' | 'SB' | 'BB';
export type TableSeats = 2 | 6 | 9;

export interface TableFormat {
  seats: TableSeats;
  label: string;
  positions: Position[];
}

export const HEADS_UP: TableFormat = {
  seats: 2,
  label: 'Heads up',
  positions: ['BTN', 'BB'],
};

export const SIX_MAX: TableFormat = {
  seats: 6,
  label: '6-max',
  positions: ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB'],
};

export const FULL_RING: TableFormat = {
  seats: 9,
  label: '9-max',
  positions: ['UTG', 'UTG1', 'MP', 'LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB'],
};

export const TABLE_FORMATS = [HEADS_UP, SIX_MAX, FULL_RING] as const;

export function formatForSeats(seats: TableSeats): TableFormat {
  return [HEADS_UP, SIX_MAX, FULL_RING].find(
    (format) => format.seats === seats
  ) ?? SIX_MAX;
}

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

export function positionLabelForSeats(
  position: Position,
  seats: TableSeats
): string {
  if (seats === 2 && position === 'BTN') return 'BTN / SB';
  if (seats === 9 && position === 'MP') return 'UTG+2';
  return POSITION_LABELS[position];
}

export function positionFullForSeats(
  position: Position,
  seats: TableSeats
): string {
  if (seats === 2 && position === 'BTN') return 'Button / Small Blind';
  if (seats === 9 && position === 'MP') return 'Under the Gun +2';
  return POSITION_FULL[position];
}
