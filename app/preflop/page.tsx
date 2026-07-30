'use client';

import { useMemo, useState } from 'react';
import { PreflopWorkspace } from './PreflopWorkspace';
import type { PreflopChart } from '@/data/preflop/ranges';
import { scenariosForSeats } from '@/data/preflop/catalog';
import { chartSummary, chartToStrategy } from '@/lib/preflop';
import { useSpot } from '@/lib/store';

export default function PreflopPage() {
  const { format, hero, villain, setFormat, setHero, setVillain } = useSpot();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [scenarioId, setScenarioId] = useState<string | null>(null);

  const scenarios = useMemo(
    () => scenariosForSeats(format.seats),
    [format.seats]
  );
  const scenario =
    scenarios.find((candidate) => candidate.id === scenarioId) ?? scenarios[0];

  const relevant = useMemo(
    () =>
      (scenario?.charts ?? []).filter(
        (chart) =>
          chart.formats.includes(format.seats) &&
          chart.hero === hero &&
          (chart.category === 'RFI' || chart.vs === villain)
      ),
    [format.seats, hero, scenario, villain]
  );
  const available = useMemo(
    () =>
      (scenario?.charts ?? []).filter((chart) =>
        chart.formats.includes(format.seats)
      ),
    [format.seats, scenario]
  );

  const active: PreflopChart | undefined =
    relevant.find((chart) => chart.id === selectedId) ?? relevant[0];
  const strategy = useMemo(
    () => (active ? chartToStrategy(active) : {}),
    [active]
  );
  const summary = useMemo(() => (active ? chartSummary(active) : []), [active]);

  return (
    <PreflopWorkspace
      format={format}
      hero={hero}
      villain={villain}
      scenarios={scenarios}
      scenario={scenario}
      available={available}
      active={active}
      strategy={strategy}
      summary={summary}
      onFormat={(next) => {
        setFormat(next);
        setScenarioId(null);
        setSelectedId(null);
      }}
      onMatchup={(nextHero, nextVillain) => {
        const charts = scenario?.charts ?? [];
        const response = charts.find(
          (chart) =>
            chart.formats.includes(format.seats) &&
            chart.hero === nextHero &&
            chart.category === 'vs-RFI' &&
            chart.vs === nextVillain
        );
        const opening = charts.find(
          (chart) =>
            chart.formats.includes(format.seats) &&
            chart.hero === nextHero &&
            chart.category === 'RFI'
        );
        setHero(nextHero);
        setVillain(nextVillain);
        setSelectedId((response ?? opening)?.id ?? null);
      }}
      onScenario={(id) => {
        setScenarioId(id);
        setSelectedId(null);
      }}
    />
  );
}
