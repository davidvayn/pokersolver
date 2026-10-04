'use client';

import { useEffect, useMemo, useState } from 'react';
import { ArrowRight, RefreshCw, Trash2 } from 'lucide-react';
import Link from 'next/link';
import { PageHeading } from '@/components/design/DesignShell';
import { Select } from '@/components/ui/Select';
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
  const [period, setPeriod] = useState('all');
  const [mode, setMode] = useState('all');

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

  const filtered = useMemo(() => {
    const cutoff =
      period === 'all' ? 0 : Date.now() - Number(period) * 86_400_000;
    return (
      hands?.filter(
        (hand) =>
          hand.completedAt >= cutoff && (mode === 'all' || hand.mode === mode)
      ) ?? []
    );
  }, [hands, period, mode]);
  const stats = useMemo(() => analyzePractice(filtered), [filtered]);

  async function clear() {
    if (
      !window.confirm(
        'Clear all practice history on this device? This cannot be undone.'
      )
    )
      return;
    setClearing(true);
    if (await clearPracticeHistory()) setHands([]);
    setClearing(false);
  }

  return (
    <div className="stats-study">
      <PageHeading title="Your game">
        <section className="stats-controls" aria-label="History filters">
          <Select
            label="Period"
            value={period}
            onChange={setPeriod}
            options={[
              { value: 'all', label: 'All time' },
              { value: '7', label: 'Last 7 days' },
              { value: '30', label: 'Last 30 days' },
            ]}
          />
          <Select
            label="Mode"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'all', label: 'All modes' },
              { value: 'full-hand', label: 'Full hand' },
              { value: 'preflop', label: 'Preflop' },
              { value: 'postflop', label: 'Postflop' },
              { value: 'push-fold', label: 'Push/fold' },
            ]}
          />
        </section>
        {!!hands?.length && (
          <button
            type="button"
            disabled={clearing}
            onClick={() => void clear()}
            className="design-icon-button"
            aria-label="Clear practice history"
          >
            <Trash2 className="h-5 w-5" aria-hidden="true" />
          </button>
        )}
        <Link href="/practice" className="study-button-primary">
          Practice <ArrowRight className="h-4 w-4" aria-hidden="true" />
        </Link>
      </PageHeading>

      {!hands ? (
        <div className="study-empty" role="status">
          <RefreshCw
            className="h-6 w-6 animate-spin text-accent motion-reduce:animate-none"
            aria-hidden="true"
          />
          <span>Loading history</span>
        </div>
      ) : stats.decisions > 0 ? (
        <PracticeStatsDashboard stats={stats} hands={filtered} />
      ) : (
        <>
          <section
            className="stats-empty-layout"
            aria-label="Practice overview"
          >
            {[
              ['Hands', String(filtered.length)],
              ['Decisions', '0'],
              ['Average EV loss', '—'],
              ['Strong decisions', '—'],
            ].map(([label, value]) => (
              <div key={label}>
                <span>{label}</span>
                <strong>{value}</strong>
              </div>
            ))}
          </section>
          <section className="stats-onboarding">
            <div>
              <h2>
                {hands.length
                  ? 'No matching decisions.'
                  : 'Your next hand starts here.'}
              </h2>
              <p>
                {hands.length
                  ? 'Try another period or mode.'
                  : 'Play a hand to see your progress and the spots to improve.'}
              </p>
              <Link href="/practice" className="study-button-primary">
                Deal a hand{' '}
                <ArrowRight className="h-4 w-4" aria-hidden="true" />
              </Link>
            </div>
            <div className="stats-onboarding-steps">
              {[
                ['01', 'Make a decision'],
                ['02', 'Review the strategy'],
                ['03', 'Find your next focus'],
              ].map(([number, title]) => (
                <div key={number}>
                  <strong>{number}</strong>
                  <h3>{title}</h3>
                </div>
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  );
}
