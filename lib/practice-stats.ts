import type {
  EvBreakdown,
  PracticeDecisionRecord,
  PracticeHandRecord,
  PracticeStats,
} from '@/lib/practice-types';

function mean(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function breakdown(
  decisions: PracticeDecisionRecord[],
  keyFor: (record: PracticeDecisionRecord) => string,
  labelFor: (record: PracticeDecisionRecord) => string
): EvBreakdown[] {
  const groups = new Map<
    string,
    { label: string; records: PracticeDecisionRecord[] }
  >();
  for (const record of decisions) {
    const key = keyFor(record);
    const current = groups.get(key) ?? { label: labelFor(record), records: [] };
    current.records.push(record);
    groups.set(key, current);
  }
  return [...groups.entries()]
    .map(([key, group]) => {
      const losses = group.records
        .map((record) => record.evLossBb)
        .filter((loss): loss is number => loss !== null);
      const lowConfidence = group.records.filter(
        (record) => record.lowConfidence
      ).length;
      return {
        key,
        label: group.label,
        decisions: group.records.length,
        graded: losses.length,
        averageEvLossBb: mean(losses),
        totalEvLossBb: losses.reduce((sum, loss) => sum + loss, 0),
        lowConfidencePercentage: group.records.length
          ? lowConfidence / group.records.length
          : 0,
      };
    })
    .sort(
      (first, second) =>
        (second.averageEvLossBb ?? -1) - (first.averageEvLossBb ?? -1) ||
        second.decisions - first.decisions
    );
}

function averageLoss(records: PracticeDecisionRecord[]): number | null {
  return mean(
    records
      .map((record) => record.evLossBb)
      .filter((loss): loss is number => loss !== null)
  );
}

export function analyzePractice(
  hands: PracticeHandRecord[]
): PracticeStats {
  const decisions = hands
    .flatMap((hand) => hand.decisions)
    .sort((first, second) => second.answeredAt - first.answeredAt);
  const losses = decisions
    .map((record) => record.evLossBb)
    .filter((loss): loss is number => loss !== null);
  const recent = decisions.slice(0, 50);
  const previous = decisions.slice(50, 100);
  const recentLoss = averageLoss(recent);
  const previousLoss = averageLoss(previous);
  const byStreet = breakdown(decisions, (record) => record.street, (record) => record.street);
  const byStack = breakdown(
    decisions,
    (record) => String(record.depthBb),
    (record) => `${record.depthBb}bb`
  );
  const byPosition = breakdown(
    decisions,
    (record) => record.position,
    (record) =>
      record.position === 'button-small-blind' ? 'BTN / SB' : 'Big blind'
  );
  const byAction = breakdown(
    decisions,
    (record) => record.chosenAction.kind,
    (record) => record.chosenAction.label
  );
  const byMode = breakdown(
    decisions,
    (record) => record.mode,
    (record) => record.mode.replace('-', ' ')
  );
  const bySeverity = breakdown(
    decisions,
    (record) => record.grade,
    (record) => record.grade
  );
  const lowConfidence = decisions.filter((record) => record.lowConfidence).length;
  const weaknesses = breakdown(
    decisions,
    (record) =>
      [record.street, record.position, record.depthBb, record.handBucket, record.facingAction].join('|'),
    (record) =>
      `${record.street} · ${record.position === 'button-small-blind' ? 'BTN / SB' : 'BB'} · ${record.handBucket} · ${record.facingAction}`
  )
    .filter((item) => item.decisions >= 2)
    .slice(0, 6);

  return {
    hands: hands.length,
    decisions: decisions.length,
    gradedDecisions: losses.length,
    averageEvLossBb: mean(losses),
    totalEvLossBb: losses.reduce((sum, loss) => sum + loss, 0),
    lowConfidencePercentage: decisions.length
      ? lowConfidence / decisions.length
      : 0,
    averageResponseMs: decisions.length
      ? decisions.reduce((sum, record) => sum + record.responseMs, 0) /
        decisions.length
      : 0,
    trendEvLossBb:
      recent.length === 50 &&
      previous.length === 50 &&
      recentLoss !== null &&
      previousLoss !== null
        ? recentLoss - previousLoss
        : null,
    byStreet,
    byStack,
    byPosition,
    byAction,
    byMode,
    bySeverity,
    weaknesses,
    recentCostly: decisions
      .filter((record) => record.evLossBb !== null)
      .sort((first, second) => (second.evLossBb ?? 0) - (first.evLossBb ?? 0))
      .slice(0, 12),
  };
}

export type {
  EvBreakdown,
  PracticeDecisionRecord,
  PracticeHandRecord,
  PracticeStats,
} from '@/lib/practice-types';
