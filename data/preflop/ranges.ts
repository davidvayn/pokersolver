// Bundled preflop range charts (approximate GTO, 100bb 6-max cash).
//
// These are a curated seed set intended for study; they are editable and the
// schema is built to extend to more formats, stack depths, and scenarios.
// Each action carries a range string (PioSolver notation) and a color; the
// chart viewer renders every hand class as stacked frequency bars, with the
// remainder implicitly "fold".

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

// --- BB defense vs a Button open (mixed call / 3-bet) -----------------------

const BB_VS_BTN = {
  threebet: 'JJ+,AQs+,AKo,A5s,A4s,KJs',
  call: '22-TT,A2s-AJs,K9s+,Q9s+,J9s+,T8s+,97s+,86s+,76s,65s,54s,ATo-AQo,KTo+,QTo+,JTo',
};

// --- vs UTG open (CO cold-call / 3-bet), a tighter example ------------------

const CO_VS_UTG = {
  threebet: 'QQ+,AKs,AKo,A5s',
  call: 'TT-99,AQs,AJs,KQs',
};

export const CHARTS: PreflopChart[] = [
  ...(Object.keys(RFI) as Position[]).map((hero) => ({
    id: `rfi-${hero}`,
    title: `${hero} — Raise First In`,
    hero,
    category: 'RFI' as const,
    actions: [{ name: 'Raise', color: RAISE, range: RFI[hero] }],
  })),
  {
    id: 'bb-vs-btn',
    title: 'BB vs BTN open',
    hero: 'BB',
    vs: 'BTN',
    category: 'vs-RFI',
    actions: [
      { name: '3-bet', color: THREEBET, range: BB_VS_BTN.threebet },
      { name: 'Call', color: CALL, range: BB_VS_BTN.call },
    ],
  },
  {
    id: 'co-vs-utg',
    title: 'CO vs UTG open',
    hero: 'CO',
    vs: 'UTG',
    category: 'vs-RFI',
    actions: [
      { name: '3-bet', color: THREEBET, range: CO_VS_UTG.threebet },
      { name: 'Call', color: CALL, range: CO_VS_UTG.call },
    ],
  },
];

export function chartsFor(hero: Position, vs?: Position): PreflopChart[] {
  return CHARTS.filter(
    (c) => c.hero === hero && (vs ? c.vs === vs : true)
  );
}
