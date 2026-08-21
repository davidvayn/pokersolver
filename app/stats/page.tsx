'use client';

import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Clock3,
  Database,
  Gauge,
  Layers3,
  RefreshCw,
  ShieldAlert,
  Target,
  Trash2,
  TrendingDown,
} from 'lucide-react';
import { cardToStr } from '@/lib/cards';
import {
  clearPracticeHistory,
  loadPracticeHands,
  subscribePracticeHistory,
} from '@/lib/practice-history';
import { analyzePractice } from '@/lib/practice-stats';
import type {
  EvBreakdown,
  PracticeDecisionRecord,
  PracticeHandRecord,
} from '@/lib/practice-types';

function bb(value: number | null): string {
  return value === null ? '—' : `${value.toFixed(3)}bb`;
}

function duration(value: number): string {
  if (!value) return '—';
  return value < 1_000 ? `${Math.round(value)}ms` : `${(value / 1_000).toFixed(1)}s`;
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function SummaryCard({
  label,
  value,
  detail,
  icon: Icon,
}: {
  label: string;
  value: string;
  detail: string;
  icon: typeof Gauge;
}) {
  return (
    <div className="border-t border-border py-4 sm:p-5 sm:first:border-l-0">
      <div className="flex items-center justify-between gap-3 text-sm text-muted">
        <span>{label}</span>
        <Icon className="h-4 w-4 text-accent" aria-hidden="true" />
      </div>
      <p className="mt-3 font-mono text-2xl font-semibold">{value}</p>
      <p className="mt-1 text-xs leading-5 text-muted">{detail}</p>
    </div>
  );
}

function Breakdown({
  title,
  items,
}: {
  title: string;
  items: EvBreakdown[];
}) {
  return (
    <section className="overflow-hidden rounded-lg border border-border bg-surface">
      <div className="border-b border-border px-4 py-3">
        <h2 className="text-sm font-semibold">{title}</h2>
      </div>
      {items.length === 0 ? (
        <p className="p-4 text-sm text-muted">No decisions in this breakdown yet.</p>
      ) : (
        <div className="divide-y divide-border">
          {items.slice(0, 8).map((item) => {
            const width = Math.min(100, (item.averageEvLossBb ?? 0) * 200);
            return (
              <div key={item.key} className="p-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium capitalize">{item.label}</p>
                    <p className="mt-1 text-xs text-muted">
                      {item.decisions} decisions · {item.graded} graded
                    </p>
                  </div>
                  <div className="shrink-0 text-right">
                    <p className="font-mono text-sm font-semibold">{bb(item.averageEvLossBb)}</p>
                    <p className="mt-1 text-[11px] text-muted">{percent(item.lowConfidencePercentage)} low confidence</p>
                  </div>
                </div>
                <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-surface-2">
                  <div
                    className="h-full rounded-full bg-raise"
                    style={{ width: `${Math.max(item.graded ? 1 : 0, width)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

function CostlyDecision({
  decision,
  hand,
}: {
  decision: PracticeDecisionRecord;
  hand?: PracticeHandRecord;
}) {
  return (
    <details className="group border-b border-border last:border-b-0">
      <summary className="flex min-h-16 cursor-pointer list-none items-center justify-between gap-4 px-4 py-3 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent sm:px-5">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold">
            {decision.handBucket} · {decision.street} · {decision.chosenAction.label}
          </p>
          <p className="mt-1 font-mono text-xs text-muted">
            {decision.heroCards.map(cardToStr).join(' ')} · {decision.depthBb}bb · {decision.position === 'button-small-blind' ? 'BTN / SB' : 'BB'}
          </p>
        </div>
        <div className="shrink-0 text-right">
          <p className="font-mono text-sm font-semibold text-raise">{bb(decision.evLossBb)}</p>
          <p className="mt-1 text-[11px] capitalize text-muted">{decision.grade}</p>
        </div>
      </summary>
      <div className="border-t border-border bg-surface-2/45 px-4 py-4 text-xs sm:px-5">
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <p className="font-semibold">Complete line</p>
            <ol className="mt-2 space-y-1 text-muted">
              {hand?.actions.map((action) => (
                <li key={action.id}>
                  {action.actor === hand.hero ? 'Hero' : 'Opponent'} · {action.street} · {action.label}
                </li>
              )) ?? <li>Action history unavailable</li>}
            </ol>
            {hand && (
              <p className="mt-3 font-mono text-muted">
                Board {hand.board.length ? hand.board.map(cardToStr).join(' ') : '—'} · Opponent {hand.opponentCards.map(cardToStr).join(' ')}
              </p>
            )}
          </div>
          <div>
            <p className="font-semibold">Policy mix</p>
            <div className="mt-2 space-y-2">
              {decision.policyActions.map((action) => (
                <div key={action.id} className="flex justify-between gap-3 text-muted">
                  <span>{action.label}</span>
                  <span className="font-mono">{percent(action.probability)} · {action.evBb === null ? 'EV unavailable' : `${action.evBb.toFixed(3)}bb`}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        {decision.lowConfidence && (
          <p className="mt-4 flex gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 p-3 leading-5">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            This record carries a {decision.confidence} confidence warning for its EV estimate. The frequency grade uses the frozen policy mix.
          </p>
        )}
      </div>
    </details>
  );
}

export default function StatsPage() {
  const [hands, setHands] = useState<PracticeHandRecord[] | null>(null);
  const [clearing, setClearing] = useState(false);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      const records = await loadPracticeHands();
      if (active) setHands(records);
    };
    void refresh();
    const unsubscribe = subscribePracticeHistory(() => void refresh());
    return () => {
      active = false;
      unsubscribe();
    };
  }, []);

  const stats = useMemo(() => (hands ? analyzePractice(hands) : null), [hands]);
  const handById = useMemo(
    () => new Map((hands ?? []).map((hand) => [hand.id, hand])),
    [hands]
  );

  async function clear() {
    if (!window.confirm('Clear the new IndexedDB practice history on this device? This cannot be undone.')) return;
    setClearing(true);
    if (await clearPracticeHistory()) setHands([]);
    setClearing(false);
  }

  if (!stats || !hands) {
    return (
      <div className="grid min-h-[50vh] place-items-center" role="status">
        <RefreshCw className="h-6 w-6 animate-spin text-accent motion-reduce:animate-none" aria-hidden="true" />
        <span className="sr-only">Loading IndexedDB practice history</span>
      </div>
    );
  }

  return (
    <div className="pb-10">
      <header className="flex flex-wrap items-end justify-between gap-4 border-b border-border pb-5">
        <div>
          <div className="flex items-center gap-2 font-mono text-xs font-semibold uppercase text-accent">
            <BarChart3 className="h-4 w-4" aria-hidden="true" />
            Practice analysis
          </div>
          <h1 className="mt-3 text-3xl font-semibold">EV-loss stats</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-muted">
            Built only from the fresh IndexedDB hand schema. Unused legacy localStorage history is intentionally ignored.
          </p>
        </div>
        <div className="flex gap-2">
          {hands.length > 0 && (
            <button
              type="button"
              disabled={clearing}
              onClick={() => void clear()}
              className="inline-flex min-h-11 items-center gap-2 rounded-md border border-border px-4 text-sm font-semibold text-muted hover:border-raise hover:text-raise focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:opacity-50"
            >
              <Trash2 className="h-4 w-4" aria-hidden="true" />
              Clear history
            </button>
          )}
          <Link
            href="/practice"
            className="inline-flex min-h-11 items-center gap-2 rounded-md bg-accent px-4 text-sm font-semibold text-accent-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            Practice
            <ArrowRight className="h-4 w-4" aria-hidden="true" />
          </Link>
        </div>
      </header>

      {stats.decisions === 0 ? (
        <section className="mt-8 rounded-lg border border-dashed border-border p-10 text-center">
          <Database className="mx-auto h-6 w-6 text-accent" aria-hidden="true" />
          <h2 className="mt-3 text-lg font-semibold">No new-format decisions yet</h2>
          <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-muted">
            Complete a decision on the heads-up table. Full-hand depths remain unavailable until validated; push/fold is ready now.
          </p>
        </section>
      ) : (
        <>
          <section className="mt-6 grid border-y border-border sm:grid-cols-2 sm:divide-x sm:divide-border xl:grid-cols-4">
            <SummaryCard icon={Gauge} label="Average EV loss" value={bb(stats.averageEvLossBb)} detail={`${stats.gradedDecisions} of ${stats.decisions} decisions graded`} />
            <SummaryCard icon={TrendingDown} label="Total EV loss" value={stats.gradedDecisions ? bb(stats.totalEvLossBb) : '—'} detail={stats.trendEvLossBb === null ? 'Need 100 decisions for trend' : `${stats.trendEvLossBb > 0 ? '+' : ''}${stats.trendEvLossBb.toFixed(3)}bb recent vs previous`} />
            <SummaryCard icon={ShieldAlert} label="Low confidence" value={percent(stats.lowConfidencePercentage)} detail="Includes approximate or unavailable action-EV estimates" />
            <SummaryCard icon={Clock3} label="Response time" value={duration(stats.averageResponseMs)} detail={`${stats.hands} complete hands retained`} />
          </section>

          {stats.gradedDecisions === 0 && (
            <div className="mt-5 flex gap-3 rounded-md border border-amber-500/30 bg-amber-500/10 p-4 text-sm leading-6">
              <AlertTriangle className="mt-1 h-4 w-4 shrink-0" aria-hidden="true" />
            <p>Push/fold action EVs use deterministic sampled showdown equity and a conservative error bound. Their EV-loss grades remain explicitly low confidence.</p>
            </div>
          )}

          <div className="mt-6 grid gap-5 lg:grid-cols-2 xl:grid-cols-3">
            <Breakdown title="By street" items={stats.byStreet} />
            <Breakdown title="By stack" items={stats.byStack} />
            <Breakdown title="By position" items={stats.byPosition} />
            <Breakdown title="By chosen action" items={stats.byAction} />
            <Breakdown title="By mode" items={stats.byMode} />
            <Breakdown title="By severity" items={stats.bySeverity} />
          </div>

          <div className="mt-6 grid gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">
            <section className="overflow-hidden rounded-lg border border-border bg-surface">
              <div className="border-b border-border px-4 py-4 sm:px-5">
                <h2 className="text-sm font-semibold">Recent costly decisions</h2>
                <p className="mt-1 text-xs text-muted">Expand a row for the complete cards, line, policy mix, and confidence.</p>
              </div>
              {stats.recentCostly.length > 0 ? (
                stats.recentCostly.map((decision) => (
                  <CostlyDecision key={decision.id} decision={decision} hand={handById.get(decision.handId)} />
                ))
              ) : (
                <p className="p-5 text-sm text-muted">No graded costly decisions yet.</p>
              )}
            </section>

            <section className="rounded-lg border border-border bg-surface p-4">
              <div className="flex items-center gap-2">
                <Target className="h-4 w-4 text-accent" aria-hidden="true" />
                <h2 className="text-sm font-semibold">Adaptive weaknesses</h2>
              </div>
              <p className="mt-2 text-xs leading-5 text-muted">The adaptive sampler considers the latest 200 graded decisions and keeps 30% authentic random coverage.</p>
              <ol className="mt-4 space-y-3">
                {stats.weaknesses.length ? stats.weaknesses.map((item) => (
                  <li key={item.key} className="border-l-2 border-raise pl-3">
                    <p className="text-xs font-semibold capitalize">{item.label}</p>
                    <p className="mt-1 font-mono text-[11px] text-muted">{bb(item.averageEvLossBb)} average · {item.decisions} attempts</p>
                  </li>
                )) : <li className="text-xs text-muted">Need at least two decisions in a group.</li>}
              </ol>
            </section>
          </div>
        </>
      )}
    </div>
  );
}
