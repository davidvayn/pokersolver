'use client';

import type { EquityResult } from '@/lib/equity/compute';

interface EquityPanelProps {
  result: EquityResult | null;
  labels: string[];
  colors: string[];
  running?: boolean;
}

export function EquityPanel({ result, labels, colors, running }: EquityPanelProps) {
  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold">Equity</h3>
        {result && (
          <span className="text-xs text-muted">
            {result.exact ? 'Exact' : 'Monte Carlo'} ·{' '}
            {result.samples.toLocaleString()} boards
          </span>
        )}
      </div>

      {!result ? (
        <div className="grid h-24 place-items-center text-sm text-muted">
          {running ? 'Calculating…' : 'Set hands and calculate'}
        </div>
      ) : (
        <div className="flex flex-col gap-4">
          {result.equities.map((eq, i) => (
            <div key={i}>
              <div className="mb-1 flex items-center justify-between text-sm">
                <span className="flex items-center gap-2">
                  <span
                    className="h-3 w-3 rounded-full"
                    style={{ background: colors[i] }}
                  />
                  {labels[i]}
                </span>
                <span className="font-mono font-semibold tabular-nums">
                  {(eq * 100).toFixed(2)}%
                </span>
              </div>
              <div className="h-3 overflow-hidden rounded-full bg-surface-2">
                <div
                  className="h-full rounded-full transition-[width] duration-300"
                  style={{ width: `${eq * 100}%`, background: colors[i] }}
                />
              </div>
              <div className="mt-1 text-[11px] text-muted">
                win {(result.wins[i] * 100).toFixed(1)}% · tie{' '}
                {(result.ties[i] * 100).toFixed(1)}%
              </div>
            </div>
          ))}

          {result.equities.length === 2 && (
            <PotOdds heroEquity={result.equities[0]} />
          )}
        </div>
      )}
    </div>
  );
}

/** Shows the pot odds / required price implied by hero equity. */
function PotOdds({ heroEquity }: { heroEquity: number }) {
  // With equity E, hero breaks even calling an amount up to E of the FINAL pot
  // (call C into final pot P where E = C/P). Equivalently the pot must lay at
  // least (1-E):E odds.
  const breakEvenPct = heroEquity * 100; // max call as % of final pot
  const ratio = heroEquity > 0 ? (1 - heroEquity) / heroEquity : Infinity;
  return (
    <div className="rounded-md border border-border bg-surface-2 p-3 text-xs">
      <div className="mb-1 font-medium">Pot odds (hero)</div>
      <div className="flex justify-between text-muted">
        <span>Break-even call</span>
        <span className="font-mono text-fg">
          {breakEvenPct.toFixed(1)}% of final pot
        </span>
      </div>
      <div className="flex justify-between text-muted">
        <span>Min. pot odds</span>
        <span className="font-mono text-fg">
          {isFinite(ratio) ? `${ratio.toFixed(2)} : 1` : '—'}
        </span>
      </div>
    </div>
  );
}
