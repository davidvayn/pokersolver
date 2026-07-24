'use client';

import { useMemo, useState } from 'react';
import { HandMatrix } from '@/components/hand-matrix/HandMatrix';
import { PokerTable } from '@/components/table/PokerTable';
import { CHARTS, type PreflopChart } from '@/data/preflop/ranges';
import { chartSummary, chartToStrategy } from '@/lib/preflop';
import { positionFullForSeats, TABLE_FORMATS } from '@/lib/positions';
import { useSpot } from '@/lib/store';

export default function PreflopPage() {
  const { format, hero, villain, setFormat, setHero, setVillain } = useSpot();
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const relevant = useMemo(
    () =>
      CHARTS.filter(
        (chart) =>
          chart.formats.includes(format.seats) &&
          chart.hero === hero &&
          (chart.category === 'RFI' || chart.vs === villain)
      ),
    [format.seats, hero, villain]
  );

  const active: PreflopChart | undefined =
    relevant.find((chart) => chart.id === selectedId) ?? relevant[0];
  const strategy = useMemo(
    () => (active ? chartToStrategy(active) : {}),
    [active]
  );
  const summary = useMemo(() => (active ? chartSummary(active) : []), [active]);

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-4 border-b border-border pb-5 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="mb-1 font-mono text-xs font-semibold uppercase text-accent">
            Preflop library
          </p>
          <h1 className="text-2xl font-semibold">Opening and response ranges</h1>
        </div>
        <div
          className="inline-flex w-fit rounded-md border border-border bg-surface p-1"
          aria-label="Table format"
        >
          {TABLE_FORMATS.map((option) => (
            <button
              key={option.seats}
              type="button"
              aria-pressed={format.seats === option.seats}
              onClick={() => setFormat(option)}
              className={
                'min-h-11 rounded px-3 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ' +
                (format.seats === option.seats
                  ? 'bg-fg text-bg'
                  : 'text-muted hover:bg-surface-2 hover:text-fg')
              }
            >
              {option.label}
            </button>
          ))}
        </div>
      </header>

      <div className="grid gap-6 lg:grid-cols-[380px_minmax(0,1fr)]">
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
            <div className="flex flex-wrap gap-2">
              {relevant.map((chart) => (
                <button
                  key={chart.id}
                  type="button"
                  onClick={() => setSelectedId(chart.id)}
                  aria-pressed={active?.id === chart.id}
                  className={
                    'min-h-10 rounded-md border px-3 py-2 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ' +
                    (active?.id === chart.id
                      ? 'border-accent bg-accent/10 text-fg'
                      : 'border-border text-muted hover:text-fg')
                  }
                >
                  {chart.category === 'RFI' ? 'Raise first in' : `vs ${chart.vs}`}
                </button>
              ))}
            </div>
          )}
        </div>

        <section className="rounded-lg border border-border bg-surface p-4 sm:p-5">
          {active ? (
            <>
              <div className="mb-4 flex flex-col gap-3 border-b border-border pb-4 sm:flex-row sm:items-end sm:justify-between">
                <div>
                  <h2 className="font-semibold">{active.title}</h2>
                  <p className="text-sm text-muted">
                    {positionFullForSeats(active.hero, format.seats)}
                    {active.vs
                      ? ` vs ${positionFullForSeats(active.vs, format.seats)}`
                      : ''}
                  </p>
                </div>
                <div className="flex flex-wrap gap-x-4 gap-y-2 text-xs">
                  {summary.map((item) => (
                    <span key={item.name} className="flex items-center gap-2">
                      <span
                        className="h-3 w-3 rounded-sm"
                        style={{ background: item.color }}
                        aria-hidden
                      />
                      <span className="text-muted">
                        {item.name}{' '}
                        <strong className="font-mono text-fg">
                          {item.pct.toFixed(0)}%
                        </strong>
                      </span>
                    </span>
                  ))}
                </div>
              </div>
              <div className="mx-auto max-w-2xl">
                <HandMatrix mode="display" strategy={strategy} />
              </div>
            </>
          ) : (
            <div className="grid min-h-80 place-items-center text-center text-sm text-muted">
              <div>
                <p className="font-medium text-fg">No chart for this matchup</p>
                <p className="mt-1">Choose another hero or opposing seat.</p>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
