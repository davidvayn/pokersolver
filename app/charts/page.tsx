'use client';

import { useMemo, useState } from 'react';
import { useSpot } from '@/lib/store';
import { PokerTable } from '@/components/table/PokerTable';
import { HandMatrix } from '@/components/hand-matrix/HandMatrix';
import { CHARTS, PreflopChart } from '@/data/preflop/ranges';
import { chartToStrategy, chartSummary } from '@/lib/preflop';
import { POSITION_FULL } from '@/lib/positions';

export default function ChartsPage() {
  const { format, hero, villain, setHero, setVillain, toggleFormat } = useSpot();
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // Charts relevant to the current hero (and villain, if a vs-chart exists).
  const relevant = useMemo(() => {
    const heroCharts = CHARTS.filter((c) => c.hero === hero);
    const vsMatch = heroCharts.filter((c) => c.vs === villain);
    const rfi = heroCharts.filter((c) => c.category === 'RFI');
    return [...vsMatch, ...rfi];
  }, [hero, villain]);

  const active: PreflopChart | undefined =
    relevant.find((c) => c.id === selectedId) ?? relevant[0];

  const strategy = useMemo(
    () => (active ? chartToStrategy(active) : {}),
    [active]
  );
  const summary = useMemo(() => (active ? chartSummary(active) : []), [active]);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-xl font-semibold">Preflop GTO Charts</h1>
          <p className="text-sm text-muted">
            Click a seat or press 1–{format.positions.length} to switch
            positions instantly.
          </p>
        </div>
        <button
          onClick={toggleFormat}
          className="rounded-md border border-border px-3 py-1.5 text-sm text-muted hover:text-fg"
        >
          {format.seats}-max ↔ {format.seats === 6 ? '9' : '6'}-max
        </button>
      </div>

      <div className="grid gap-6 lg:grid-cols-[380px_1fr]">
        <div className="flex flex-col gap-4">
          <div className="rounded-lg border border-border bg-surface p-4">
            <PokerTable
              format={format}
              hero={hero}
              villain={villain}
              onHero={setHero}
              onVillain={setVillain}
            />
          </div>

          {relevant.length > 1 && (
            <div className="flex flex-wrap gap-1">
              {relevant.map((c) => (
                <button
                  key={c.id}
                  onClick={() => setSelectedId(c.id)}
                  className={
                    'rounded-md border px-3 py-1.5 text-xs transition-colors ' +
                    (active?.id === c.id
                      ? 'border-accent text-fg'
                      : 'border-border text-muted hover:text-fg')
                  }
                >
                  {c.title}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="rounded-lg border border-border bg-surface p-4">
          {active ? (
            <>
              <div className="mb-3 flex items-center justify-between">
                <div>
                  <div className="font-semibold">{active.title}</div>
                  <div className="text-xs text-muted">
                    {POSITION_FULL[active.hero]}
                    {active.vs ? ` vs ${POSITION_FULL[active.vs]}` : ''}
                  </div>
                </div>
                <div className="flex flex-wrap justify-end gap-3 text-xs">
                  {summary.map((s) => (
                    <span key={s.name} className="flex items-center gap-1.5">
                      <span
                        className="h-3 w-3 rounded-sm"
                        style={{ background: s.color }}
                      />
                      {s.name} {s.pct.toFixed(0)}%
                    </span>
                  ))}
                </div>
              </div>
              <div className="mx-auto max-w-xl">
                <HandMatrix mode="display" strategy={strategy} />
              </div>
            </>
          ) : (
            <div className="grid h-64 place-items-center text-center text-sm text-muted">
              <div>
                <p>No bundled chart for {hero}{villain ? ` vs ${villain}` : ''} yet.</p>
                <p className="mt-1">
                  Try BTN / CO / UTG (RFI), or BB vs BTN, CO vs UTG.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
