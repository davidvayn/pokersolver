import {
  comboLabelToCombos,
  comboKey,
  parseRange,
  handClassLabel,
} from './cards';
import type { StrategySegment } from '@/components/hand-matrix/HandMatrix';
import type { PreflopChart } from '@/data/preflop/ranges';

const FOLD_COLOR = 'rgb(var(--fold))';

/**
 * Turn a chart into per-hand-class stacked strategy segments. For each of the
 * 169 classes we compute what fraction of its combos falls into each action;
 * the leftover fraction is shown as "fold".
 */
export function chartToStrategy(
  chart: PreflopChart
): Record<string, StrategySegment[]> {
  const parsed = chart.actions.map((a) => ({
    ...a,
    range: parseRange(a.range),
  }));

  const out: Record<string, StrategySegment[]> = {};

  for (let row = 0; row < 13; row++) {
    for (let col = 0; col < 13; col++) {
      const label = handClassLabel(row, col);
      const combos = comboLabelToCombos(label);
      const total = combos.length;
      let used = 0;
      const segs: StrategySegment[] = [];

      for (const action of parsed) {
        let inAction = 0;
        for (const [a, b] of combos) {
          const w = action.range.get(comboKey(a, b)) ?? 0;
          inAction += Math.min(1, w);
        }
        const frac = inAction / total;
        if (frac > 0.001) {
          segs.push({ color: action.color, fraction: frac, label: action.name });
          used += frac;
        }
      }

      const fold = Math.max(0, 1 - used);
      if (fold > 0.001 && segs.length > 0) {
        segs.push({ color: FOLD_COLOR, fraction: fold, label: 'Fold' });
      }
      if (segs.length) out[label] = segs;
    }
  }
  return out;
}

/** Overall action frequencies across the whole 1326-combo space. */
export function chartSummary(
  chart: PreflopChart
): { name: string; color: string; pct: number }[] {
  const parsed = chart.actions.map((a) => ({
    ...a,
    range: parseRange(a.range),
  }));
  let acted = 0;
  const rows: { name: string; color: string; pct: number }[] = parsed.map((action) => {
    let combos = 0;
    for (const w of action.range.values()) combos += Math.min(1, w);
    acted += combos;
    return { name: action.name, color: action.color, pct: (combos / 1326) * 100 };
  });
  rows.push({
    name: 'Fold',
    color: FOLD_COLOR,
    pct: Math.max(0, ((1326 - acted) / 1326) * 100),
  });
  return rows;
}
