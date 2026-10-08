import type { CashGameRules } from '@/lib/cash-game-rules';
import type { CashSettlement } from '@/lib/cash-settlement';

export function CashSettlementSummary({ settlement, rules }: { settlement: CashSettlement; rules: CashGameRules }) {
  const amount = (units: number) => `${(units / rules.unitsPerBb).toFixed(2)}bb`;
  const rows = [
    ['Gross contestable pot', settlement.grossPotUnits],
    ['House rake', settlement.rakeUnits],
    ['Net awarded pot', settlement.netPotUnits],
    ['Uncalled wagers returned', settlement.uncalledRefundUnits[0] + settlement.uncalledRefundUnits[1]],
  ] as const;
  return (
    <dl aria-label="Cash hand settlement" className="grid grid-cols-2 gap-3 rounded-md border border-border bg-surface p-3 text-xs">
      {rows.map(([label, value]) => <div key={label}>
        <dt className="leading-5 text-muted">{label}</dt>
        <dd className="mt-1 font-mono font-semibold tabular-nums">{amount(value)}</dd>
      </div>)}
    </dl>
  );
}
