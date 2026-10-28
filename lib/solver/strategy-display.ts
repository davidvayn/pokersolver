import type { NodeStrategy } from './client';

export function solverActionLabel(label: string): string {
  return label
    .replace(/\bto\s+/g, '')
    .replace(/(\d+(?:\.\d+)?)\s*bb\b/g, (_, amount: string) =>
      String(Number(amount))
    );
}

/** A display filter: retain original probabilities, EVs, and the source policy. */
export function compactStrategy(node: NodeStrategy): NodeStrategy {
  const sizes = node.actions.filter(
    (label) => /^(Bet|Raise)/.test(label) && !/all.in/i.test(label)
  );
  const amount = (label: string) =>
    Number(label.match(/\d+(?:\.\d+)?/)?.[0] ?? 0);
  const largest = sizes.reduce<string | undefined>(
    (best, label) => (!best || amount(label) > amount(best) ? label : best),
    undefined
  );
  const shown = new Set(
    node.actions.filter((label) => !sizes.includes(label) || label === largest)
  );
  const displayActions = (actions: NodeStrategy['rows'][number]['actions']) =>
    actions
      .filter((action) => shown.has(action.action))
      .map((action) => ({
        ...action,
        action: solverActionLabel(action.action),
      }));
  return {
    ...node,
    actions: node.actions
      .filter((label) => shown.has(label))
      .map(solverActionLabel),
    rows: node.rows.map((row) => ({
      ...row,
      actions: displayActions(row.actions),
      combos_data: row.combos_data?.map((combo) => ({
        ...combo,
        actions: displayActions(combo.actions),
      })),
    })),
  };
}
