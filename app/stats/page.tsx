'use client';

import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';
import {
  ArrowRight,
  BarChart3,
  Database,
  RefreshCw,
  Sparkles,
  Trash2,
} from 'lucide-react';
import { PracticeStatsDashboard } from '@/components/stats/PracticeStatsDashboard';
import {
  clearPracticeHistory,
  loadPracticeHands,
  subscribePracticeHistory,
} from '@/lib/practice-history';
import { analyzePractice } from '@/lib/practice-stats';
import type { PracticeHandRecord } from '@/lib/practice-types';

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

  async function clear() {
    if (!window.confirm('Clear all practice history on this device? This cannot be undone.')) return;
    setClearing(true);
    if (await clearPracticeHistory()) setHands([]);
    setClearing(false);
  }

  if (!stats || !hands) {
    return (
      <div className="grid min-h-[50vh] place-items-center" role="status">
        <RefreshCw className="h-6 w-6 animate-spin text-accent motion-reduce:animate-none" aria-hidden="true" />
        <span className="sr-only">Loading practice history</span>
      </div>
    );
  }

  return (
    <div className="pb-10">
      <header className="stats-page-header flex flex-wrap items-end justify-between gap-5 border-b border-border pb-5">
        <div>
          <div className="flex items-center gap-2 font-mono text-xs font-semibold uppercase text-accent">
            <BarChart3 className="h-4 w-4" aria-hidden="true" />
            Practice intelligence
            {hands.length > 0 && (
              <span className="ml-1 inline-flex items-center gap-1.5 rounded-full bg-accent/10 px-2 py-1 normal-case text-accent">
                <span className="stats-live-dot h-1.5 w-1.5 rounded-full bg-accent" aria-hidden="true" />
                Live
              </span>
            )}
          </div>
          <h1 className="mt-3 text-3xl font-semibold tracking-tight sm:text-4xl">
            See the shape of your game.
          </h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-muted">
            Every view is built from the same local decisions and frequency grades used at the Practice table.
          </p>
        </div>
        <div className="flex gap-2">
          {hands.length > 0 && (
            <button
              type="button"
              disabled={clearing}
              onClick={() => void clear()}
              className="inline-flex min-h-11 items-center gap-2 rounded-md border border-border bg-surface px-4 text-sm font-semibold text-muted transition-colors hover:border-raise hover:text-raise focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:opacity-50"
            >
              <Trash2 className="h-4 w-4" aria-hidden="true" />
              Clear
            </button>
          )}
          <Link
            href="/practice"
            className="inline-flex min-h-11 items-center gap-2 rounded-md bg-accent px-4 text-sm font-semibold text-accent-fg transition-opacity hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            Practice
            <ArrowRight className="h-4 w-4" aria-hidden="true" />
          </Link>
        </div>
      </header>

      {stats.decisions === 0 ? (
        <section className="stats-empty-state relative mt-8 overflow-hidden rounded-lg border border-dashed border-border bg-surface px-5 py-16 text-center sm:py-20">
          <div className="pointer-events-none absolute inset-0 opacity-60" aria-hidden="true" />
          <div className="relative">
            <span className="mx-auto grid h-12 w-12 place-content-center rounded-lg border border-border bg-bg shadow-sm">
              <Database className="h-5 w-5 text-accent" aria-hidden="true" />
            </span>
            <h2 className="mt-4 text-lg font-semibold">Your dashboard starts with one decision</h2>
            <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-muted">
              Finish a practice hand to unlock trends, activity, quality, speed, and spot-level weaknesses.
            </p>
            <Link
              href="/practice"
              className="mt-6 inline-flex min-h-11 items-center gap-2 rounded-md bg-accent px-4 text-sm font-semibold text-accent-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            >
              <Sparkles className="h-4 w-4" aria-hidden="true" />
              Deal a practice hand
            </Link>
          </div>
        </section>
      ) : (
        <PracticeStatsDashboard stats={stats} hands={hands} />
      )}
    </div>
  );
}
