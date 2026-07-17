// Bundled preflop range charts (approximate GTO, 100bb 6-max cash).
//
// These are a curated seed set intended for study; they are editable and the
// schema is built to extend to more formats, stack depths, and scenarios.
// Each action carries a range string (PioSolver notation) and a color; the
// chart viewer renders every hand class as stacked frequency bars, with the
// remainder implicitly "fold". Action ranges within a chart are kept disjoint
// so frequencies never sum past 100%.

import { Position } from '@/lib/positions';

export interface ChartAction {
  name: string;
  color: string;
  range: string;
}

export interface PreflopChart {
  id: string;
  title: string;
  hero: Position;
  vs?: Position;
  category: 'RFI' | 'vs-RFI';
  actions: ChartAction[];
}

const RAISE = 'rgb(var(--raise))';
const CALL = 'rgb(var(--call))';
const THREEBET = 'rgb(var(--allin))';

// --- Raise-first-in (open-raising) ranges, by position ----------------------

const RFI: Record<string, string> = {
  UTG: '77+,ATs+,KTs+,QTs+,JTs,T9s,98s,AJo+,KQo',
  MP: '66+,A9s+,KTs+,QTs+,J9s+,T9s,98s,87s,ATo+,KJo+',
  CO: '55+,A2s+,K9s+,Q9s+,J9s+,T8s+,97s+,86s+,76s,65s,ATo+,KTo+,QTo+,JTo',
  BTN: '22+,A2s+,K5s+,Q7s+,J8s+,T8s+,97s+,86s+,75s+,65s,54s,A7o+,A5o,K9o+,Q9o+,J9o+,T9o,98o',
  SB: '22+,A2s+,K6s+,Q8s+,J8s+,T8s+,97s+,86s+,76s,65s,54s,A7o+,A5o,K9o+,Q9o+,JTo',
};

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

export const CHARTS: PreflopChart[] = [
  ...(Object.keys(RFI) as Position[]).map((hero) => ({
    id: `rfi-${hero}`,
    title: `${hero} — Raise First In`,
    hero,
    category: 'RFI' as const,
    actions: [{ name: 'Raise', color: RAISE, range: RFI[hero] }],
  })),
  ...DEFENSES.map((d) => ({
    id: `${d.hero}-vs-${d.vs}`.toLowerCase(),
    title: `${d.hero} vs ${d.vs} open`,
    hero: d.hero,
    vs: d.vs,
    category: 'vs-RFI' as const,
    actions: [
      { name: '3-bet', color: THREEBET, range: d.threebet },
      { name: 'Call', color: CALL, range: d.call },
    ],
  })),
];

export function chartsFor(hero: Position, vs?: Position): PreflopChart[] {
  return CHARTS.filter(
    (c) => c.hero === hero && (vs ? c.vs === vs : true)
  );
}
