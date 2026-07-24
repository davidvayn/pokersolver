// Bundled preflop range charts (simplified 100bb cash-game baselines).
//
// These are a curated seed set intended for study; they are editable and the
// schema is built to extend to more formats, stack depths, and scenarios.
// Each action carries a range string (PioSolver notation) and a color; the
// chart viewer renders every hand class as stacked frequency bars, with the
// remainder implicitly "fold". Action ranges within a chart are kept disjoint
// so frequencies never sum past 100%.

import { POSITION_LABELS } from '@/lib/positions';
import type { Position, TableSeats } from '@/lib/positions';

export type ChartActionName = 'Raise' | 'Call' | '3-bet';

export interface ChartAction {
  name: ChartActionName;
  color: string;
  range: string;
}

export interface PreflopChart {
  id: string;
  title: string;
  hero: Position;
  vs?: Position;
  category: 'RFI' | 'vs-RFI';
  formats: TableSeats[];
  actions: ChartAction[];
}

const RAISE = 'rgb(var(--raise))';
const CALL = 'rgb(var(--call))';
const THREEBET = 'rgb(var(--allin))';

const HEADS_UP: PreflopChart[] = [
  {
    id: 'hu-btn-rfi',
    title: 'BTN / SB - Raise First In',
    hero: 'BTN',
    vs: 'BB',
    category: 'RFI',
    formats: [2],
    actions: [
      {
        name: 'Raise',
        color: RAISE,
        range:
          '22+,A2s+,K2s+,Q2s+,J2s+,T2s+,92s+,82s+,72s+,62s+,52s+,42s+,32s,A2o+,K2o+,Q5o+,J7o+,T7o+,97o+,87o,76o,65o,54o',
      },
    ],
  },
  {
    id: 'hu-bb-vs-btn',
    title: 'BB vs BTN / SB open',
    hero: 'BB',
    vs: 'BTN',
    category: 'vs-RFI',
    formats: [2],
    actions: [
      {
        name: '3-bet',
        color: THREEBET,
        range: '99+,AJs+,A5s-A2s,KQs,AQo+,KQo',
      },
      {
        name: 'Call',
        color: CALL,
        range:
          '88-22,ATs-A6s,KJs-K2s,Q2s+,J2s+,T2s+,92s+,82s+,72s+,62s+,52s+,42s+,32s,A2o-AJo,K2o-KJo,Q5o+,J7o+,T7o+,97o+,87o,76o,65o,54o',
      },
    ],
  },
];

// --- Raise-first-in (open-raising) ranges, by position ----------------------

const RFI: Record<string, string> = {
  UTG: '77+,ATs+,KTs+,QTs+,JTs,T9s,98s,AJo+,KQo',
  MP: '66+,A9s+,KTs+,QTs+,J9s+,T9s,98s,87s,ATo+,KJo+',
  CO: '55+,A2s+,K9s+,Q9s+,J9s+,T8s+,97s+,86s+,76s,65s,ATo+,KTo+,QTo+,JTo',
  BTN: '22+,A2s+,K5s+,Q7s+,J8s+,T8s+,97s+,86s+,75s+,65s,54s,A7o+,A5o,K9o+,Q9o+,J9o+,T9o,98o',
  SB: '22+,A2s+,K6s+,Q8s+,J8s+,T8s+,97s+,86s+,76s,65s,54s,A7o+,A5o,K9o+,Q9o+,JTo',
};

// Dedicated full-ring ranges. These are deliberately separate from the 6-max
// seed set: early-position ranges tighten with more players left to act.
const FULL_RING_RFI: Array<{
  hero: Exclude<Position, 'BB'>;
  range: string;
}> = [
  {
    hero: 'UTG',
    range: '77+,ATs+,KQs,QJs,JTs,T9s,AQo+',
  },
  {
    hero: 'UTG1',
    range: '66+,ATs+,KJs+,QJs,JTs,T9s,98s,AQo+,KQo',
  },
  {
    hero: 'MP',
    range: '66+,A9s+,KTs+,QTs+,JTs,T9s,98s,87s,AJo+,KQo',
  },
  {
    hero: 'LJ',
    range: '55+,A8s+,KTs+,QTs+,J9s+,T9s,98s,87s,76s,ATo+,KQo',
  },
  {
    hero: 'HJ',
    range:
      '44+,A5s+,K9s+,Q9s+,J9s+,T8s+,97s+,87s,76s,65s,ATo+,KJo+,QJo',
  },
  {
    hero: 'CO',
    range:
      '33+,A2s+,K7s+,Q9s+,J9s+,T8s+,97s+,86s+,76s,65s,54s,A8o+,KTo+,QTo+,JTo',
  },
  {
    hero: 'BTN',
    range:
      '22+,A2s+,K2s+,Q5s+,J7s+,T7s+,96s+,86s+,75s+,65s,54s,A2o+,K8o+,Q9o+,J9o+,T9o',
  },
  {
    hero: 'SB',
    range:
      '22+,A2s+,K5s+,Q7s+,J8s+,T8s+,97s+,86s+,76s,65s,54s,A7o+,K9o+,Q9o+,JTo',
  },
];

// --- Defense vs a single raise (call / 3-bet), by hero vs opener -------------
// Ranges are hand-picked to be disjoint between the two actions.

interface Defense {
  hero: Position;
  vs: Position;
  threebet: string;
  call: string;
}

const DEFENSES: Defense[] = [
  {
    hero: 'BB',
    vs: 'UTG',
    threebet: 'QQ+,AKs,AKo,A5s',
    call: 'TT-22,AJs,ATs,KQs,KJs,QJs,JTs,T9s,98s,87s,76s,65s,AQo,KQo',
  },
  {
    hero: 'BB',
    vs: 'MP',
    threebet: 'JJ+,AQs,AKs,AKo,A5s,A4s',
    call: 'TT-22,AJs,ATs,KQs,KJs,KTs,QJs,QTs,JTs,J9s,T9s,98s,87s,76s,65s,54s,AQo,KQo,KJo',
  },
  {
    hero: 'BB',
    vs: 'CO',
    threebet: 'TT+,AJs+,KQs,AQo+,A5s,A4s',
    call: '99-22,ATs,A9s,A8s,A7s,A6s,A3s,A2s,KJs,KTs,K9s,QJs,QTs,Q9s,JTs,J9s,T9s,T8s,98s,97s,87s,86s,76s,65s,54s,AJo,ATo,KJo,QJo',
  },
  {
    hero: 'BB',
    vs: 'BTN',
    threebet: 'TT+,AQs+,KJs,AJo+,A5s,A4s',
    call: '99-22,AJs,ATs,A9s,A8s,A7s,A6s,A3s,A2s,KQs,KTs,K9s,K8s,K7s,QJs,QTs,Q9s,Q8s,JTs,J9s,J8s,T9s,T8s,98s,97s,87s,86s,76s,75s,65s,64s,54s,53s,ATo,KJo,KTo,QJo,QTo,JTo',
  },
  {
    hero: 'BB',
    vs: 'SB',
    threebet: 'TT+,AJs+,KTs,KJs,KQs,QTs,QJs,JTs,AQo+,KQo,A5s,A4s,A3s',
    call: '99-22,ATs,A9s,A8s,A7s,A6s,A2s,K9s,K8s,K7s,K6s,K5s,Q9s,Q8s,J9s,J8s,T9s,T8s,98s,97s,87s,86s,76s,75s,65s,64s,54s,53s,43s,A2o-AJo,K9o,KTo,KJo,Q9o,QTo,QJo,J9o,JTo,T9o,98o',
  },
  {
    hero: 'SB',
    vs: 'BTN',
    threebet: 'TT+,AQs,AKs,AKo,KQs,KJs,A5s,A4s',
    call: '99-22,AJs,ATs,KTs,QJs,QTs,JTs,T9s,98s,AQo,AJo',
  },
  {
    hero: 'CO',
    vs: 'UTG',
    threebet: 'QQ+,AKs,AKo,A5s',
    call: 'TT-99,AQs,AJs,KQs',
  },
];

const FULL_RING_DEFENSES: Defense[] = [
  {
    hero: 'BB',
    vs: 'UTG',
    threebet: 'QQ+,AKs,AKo,A5s',
    call:
      'JJ-22,AQs-AJs,ATs,KQs,QJs,JTs,T9s,98s,87s,76s,65s,AQo,KQo',
  },
  {
    hero: 'BB',
    vs: 'UTG1',
    threebet: 'QQ+,AKs,AKo,A5s',
    call:
      'JJ-22,AQs-ATs,KQs,KJs,QJs,JTs,T9s,98s,87s,76s,65s,AQo,KQo',
  },
  {
    hero: 'BB',
    vs: 'MP',
    threebet: 'JJ+,AQs+,AKo,A5s',
    call:
      'TT-22,AJs-ATs,KQs,KJs,QJs,JTs,T9s,98s,87s,76s,65s,AQo,KQo',
  },
  {
    hero: 'BB',
    vs: 'LJ',
    threebet: 'JJ+,AQs+,AKo,A5s,A4s',
    call:
      'TT-22,AJs-ATs,KQs,KJs,KTs,QJs,QTs,JTs,T9s,98s,87s,76s,65s,54s,AQo,KQo',
  },
  {
    hero: 'BB',
    vs: 'HJ',
    threebet: 'TT+,AQs+,AKo,A5s,A4s',
    call:
      '99-22,AJs-A9s,KQs,KJs,KTs,QJs,QTs,JTs,J9s,T9s,98s,87s,76s,65s,54s,AQo,AJo,KQo',
  },
  {
    hero: 'BB',
    vs: 'CO',
    threebet: 'TT+,AJs+,KQs,AQo+,A5s,A4s',
    call:
      '99-22,ATs-A6s,A3s,A2s,KJs-K9s,QJs-Q9s,JTs,J9s,T9s,T8s,98s,97s,87s,86s,76s,65s,54s,AJo,ATo,KJo,QJo',
  },
  {
    hero: 'BB',
    vs: 'BTN',
    threebet: 'TT+,AQs+,KJs,AJo+,A5s,A4s',
    call:
      '99-22,AJs,ATs,A9s,A8s,A7s,A6s,A3s,A2s,KQs,KTs,K9s,K8s,K7s,QJs,QTs,Q9s,Q8s,JTs,J9s,J8s,T9s,T8s,98s,97s,87s,86s,76s,75s,65s,64s,54s,53s,ATo,KJo,KTo,QJo,QTo,JTo',
  },
  {
    hero: 'BB',
    vs: 'SB',
    threebet: 'TT+,AJs+,KTs,KJs,KQs,QTs,QJs,JTs,AQo+,KQo,A5s,A4s,A3s',
    call:
      '99-22,ATs,A9s,A8s,A7s,A6s,A2s,K9s,K8s,K7s,K6s,K5s,Q9s,Q8s,J9s,J8s,T9s,T8s,98s,97s,87s,86s,76s,75s,65s,64s,54s,53s,43s,A2o-AJo,K9o,KTo,KJo,Q9o,QTo,QJo,J9o,JTo,T9o,98o',
  },
];

export const CHARTS: PreflopChart[] = [
  ...HEADS_UP,
  ...(Object.keys(RFI) as Position[]).map((hero) => ({
    id: `rfi-${hero}`,
    title: `${hero} - Raise First In`,
    hero,
    category: 'RFI' as const,
    formats: [6] as TableSeats[],
    actions: [{ name: 'Raise' as const, color: RAISE, range: RFI[hero] }],
  })),
  ...DEFENSES.map((d) => ({
    id: `${d.hero}-vs-${d.vs}`.toLowerCase(),
    title: `${d.hero} vs ${d.vs} open`,
    hero: d.hero,
    vs: d.vs,
    category: 'vs-RFI' as const,
    formats: [6] as TableSeats[],
    actions: [
      { name: '3-bet' as const, color: THREEBET, range: d.threebet },
      { name: 'Call' as const, color: CALL, range: d.call },
    ],
  })),
  ...FULL_RING_RFI.map(({ hero, range }) => ({
    id: `9max-rfi-${hero.toLowerCase()}`,
    title: `${POSITION_LABELS[hero]} - Raise First In`,
    hero,
    category: 'RFI' as const,
    formats: [9] as TableSeats[],
    actions: [{ name: 'Raise' as const, color: RAISE, range }],
  })),
  ...FULL_RING_DEFENSES.map((defense) => ({
    id: `9max-${defense.hero}-vs-${defense.vs}`.toLowerCase(),
    title: `${POSITION_LABELS[defense.hero]} vs ${POSITION_LABELS[defense.vs]} open`,
    hero: defense.hero,
    vs: defense.vs,
    category: 'vs-RFI' as const,
    formats: [9] as TableSeats[],
    actions: [
      {
        name: '3-bet' as const,
        color: THREEBET,
        range: defense.threebet,
      },
      { name: 'Call' as const, color: CALL, range: defense.call },
    ],
  })),
];

export function chartsFor(
  hero: Position,
  vs?: Position,
  seats?: TableSeats
): PreflopChart[] {
  return CHARTS.filter(
    (chart) =>
      chart.hero === hero &&
      (vs ? chart.vs === vs : true) &&
      (seats ? chart.formats.includes(seats) : true)
  );
}
