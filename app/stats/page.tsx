'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import {
  ArrowRight,
  BarChart3,
  CalendarDays,
  Check,
  Clock3,
  Crosshair,
  Flame,
  Gauge,
  Layers3,
  MapPin,
  RotateCcw,
  Table2,
  Target,
  Trash2,
  TrendingDown,
  TrendingUp,
  X,
  Zap,
  type LucideIcon,
} from 'lucide-react';
import {
  clearPracticeHistory,
  loadPracticeHistory,
  subscribePracticeHistory,
} from '@/lib/practice-history';
import {
  analyzePractice,
  type PracticeRecord,
  type StatBreakdown,
} from '@/lib/practice-stats';
import { positionLabelForSeats } from '@/lib/positions';

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function formatDuration(milliseconds: number): string {
  if (milliseconds < 1000) return `${milliseconds} ms`;
  return `${(milliseconds / 1000).toFixed(1)} s`;
}

function formatTrend(value: number): string {
  const points = Math.round(Math.abs(value) * 100);
  if (points === 0) return 'No change';
  return `${value > 0 ? '+' : '-'}${points} pts`;
}

function formatAnsweredAt(timestamp: number): string {
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(timestamp);
}

function accuracyFor(records: PracticeRecord[]): number {
  if (records.length === 0) return 0;
  return records.filter((record) => record.correct).length / records.length;
}

interface SummaryCardProps {
  icon: LucideIcon;
  label: string;
  value: string;
  detail: string;
}

function SummaryCard({
  icon: Icon,
  label,
  value,
  detail,
}: SummaryCardProps) {
  return (
    <div className="rounded-lg border border-border bg-surface p-4 sm:p-5">
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm text-muted">{label}</span>
        <Icon className="h-4 w-4 text-accent" aria-hidden="true" />
      </div>
      <p className="mt-3 font-mono text-2xl font-semibold text-fg">{value}</p>
      <p className="mt-1 text-xs leading-5 text-muted">{detail}</p>
    </div>
  );
}

interface BreakdownPanelProps {
  title: string;
  description: string;
  icon: LucideIcon;
  items: StatBreakdown[];
  overallAccuracy: number;
}

function BreakdownPanel({
  title,
  description,
  icon: Icon,
  items,
  overallAccuracy,
}: BreakdownPanelProps) {
  return (
    <section
      aria-labelledby={`breakdown-${title.replace(/\s+/g, '-').toLowerCase()}`}
      className="overflow-hidden rounded-lg border border-border bg-surface"
    >
      <div className="flex items-start gap-3 border-b border-border px-4 py-4 sm:px-5">
        <div className="grid h-9 w-9 shrink-0 place-items-center rounded-md bg-surface-2 text-accent">
          <Icon className="h-4 w-4" aria-hidden="true" />
        </div>
        <div>
          <h3
            id={`breakdown-${title.replace(/\s+/g, '-').toLowerCase()}`}
            className="text-sm font-semibold text-fg"
          >
            {title}
          </h3>
          <p className="mt-0.5 text-xs leading-5 text-muted">{description}</p>
        </div>
      </div>

      <div className="divide-y divide-border">
        {items.map((item) => {
          const difference = item.accuracy - overallAccuracy;
          const assessment =
            item.attempts < 3
              ? 'Need more data'
              : Math.abs(difference) < 0.005
              ? 'At average'
              : difference > 0
                ? 'Strength'
                : 'Focus';

          return (
            <div key={item.key} className="px-4 py-3.5 sm:px-5">
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-fg">{item.label}</p>
                  <p className="mt-0.5 text-xs text-muted">
                    {item.correct} of {item.attempts} correct
                  </p>
                </div>
                <div className="shrink-0 text-right">
                  <p className="font-mono text-sm font-semibold text-fg">
                    {formatPercent(item.accuracy)}
                  </p>
                  <p
                    className={
                      'mt-0.5 text-xs font-medium ' +
                      (assessment === 'Strength'
                        ? 'text-accent'
                        : assessment === 'Focus'
                          ? 'text-raise'
                          : 'text-muted')
                    }
                  >
                    {assessment}
                  </p>
                </div>
              </div>
              <div
                className="mt-3 h-1.5 overflow-hidden rounded bg-surface-2"
                role="img"
                aria-label={`${item.label}: ${formatPercent(item.accuracy)} accuracy`}
              >
                <div
                  className={
                    'h-full rounded ' +
                    (assessment === 'Focus' ? 'bg-raise' : 'bg-accent')
                  }
                  style={{ width: `${item.accuracy * 100}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function LoadingState() {
  return (
    <div
      className="grid min-h-[320px] place-items-center rounded-lg border border-border bg-surface"
      role="status"
    >
      <div className="flex items-center gap-3 text-sm text-muted">
        <RotateCcw className="h-4 w-4 animate-spin motion-reduce:animate-none" />
        Loading practice history
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <section className="grid min-h-[440px] place-items-center border-y border-border py-16 text-center">
      <div className="max-w-md">
        <div className="mx-auto grid h-12 w-12 place-items-center rounded-lg bg-surface-2 text-accent">
          <BarChart3 className="h-6 w-6" aria-hidden="true" />
        </div>
        <h2 className="mt-5 text-xl font-semibold text-fg">No decisions recorded yet</h2>
        <p className="mt-2 text-sm leading-6 text-muted">
          Complete a practice set to establish your baseline. Your accuracy,
          timing, strengths, and review areas will appear here.
        </p>
        <Link
          href="/practice"
          className="mt-6 inline-flex min-h-11 cursor-pointer items-center justify-center gap-2 rounded-md bg-accent px-5 py-2.5 text-sm font-semibold text-accent-fg transition-opacity duration-200 hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
        >
          Start practice
          <ArrowRight className="h-4 w-4" aria-hidden="true" />
        </Link>
      </div>
    </section>
  );
}

export default function StatsPage() {
  const [records, setRecords] = useState<PracticeRecord[] | null>(null);
  const [confirmingClear, setConfirmingClear] = useState(false);

  useEffect(() => {
    const refresh = () => setRecords(loadPracticeHistory());
    refresh();
    return subscribePracticeHistory(refresh);
  }, []);

  const stats = useMemo(
    () => (records === null ? null : analyzePractice(records)),
    [records]
  );

  const comparison = useMemo(() => {
    if (!records) return null;
    const ordered = [...records].sort(
      (first, second) => second.answeredAt - first.answeredAt
    );
    const recent = ordered.slice(0, 20);
    const previous = ordered.slice(20, 40);
    return {
      recentCount: recent.length,
      previousCount: previous.length,
      recentAccuracy: accuracyFor(recent),
      previousAccuracy: accuracyFor(previous),
    };
  }, [records]);

  const handleClear = () => {
    clearPracticeHistory();
    setConfirmingClear(false);
  };

  return (
    <div className="mx-auto flex w-full max-w-[1280px] flex-col gap-7 pb-12 pt-1 sm:gap-8 sm:pt-3">
      <header className="flex flex-col gap-4 border-b border-border pb-6 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="flex items-center gap-2 font-mono text-xs font-semibold uppercase text-accent">
            <span className="h-2 w-2 rounded-full bg-accent" aria-hidden="true" />
            Practice report
          </div>
          <h1 className="mt-3 text-3xl font-bold text-fg sm:text-4xl">Stats</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-muted">
            See where your preflop decisions are reliable and what to train next.
          </p>
        </div>

        {records && records.length > 0 && !confirmingClear && (
          <button
            type="button"
            onClick={() => setConfirmingClear(true)}
            className="inline-flex min-h-11 cursor-pointer items-center justify-center gap-2 self-start rounded-md border border-border bg-surface px-4 py-2 text-sm font-medium text-muted transition-colors duration-200 hover:border-raise hover:text-raise focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-raise focus-visible:ring-offset-2 focus-visible:ring-offset-bg sm:self-auto"
          >
            <Trash2 className="h-4 w-4" aria-hidden="true" />
            Clear history
          </button>
        )}
      </header>

      {confirmingClear && records && records.length > 0 && (
        <div
          role="alertdialog"
          aria-labelledby="clear-history-title"
          aria-describedby="clear-history-description"
          className="flex flex-col gap-4 rounded-lg border border-raise/40 bg-surface p-4 sm:flex-row sm:items-center sm:justify-between sm:p-5"
        >
          <div>
            <h2 id="clear-history-title" className="text-sm font-semibold text-fg">
              Clear all practice history?
            </h2>
            <p id="clear-history-description" className="mt-1 text-sm text-muted">
              This permanently removes {records.length}{' '}
              {records.length === 1 ? 'decision' : 'decisions'} from this device.
            </p>
          </div>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setConfirmingClear(false)}
              autoFocus
              className="inline-flex min-h-11 flex-1 cursor-pointer items-center justify-center gap-2 rounded-md border border-border px-4 py-2 text-sm font-medium text-fg transition-colors duration-200 hover:bg-surface-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent sm:flex-none"
            >
              <X className="h-4 w-4" aria-hidden="true" />
              Cancel
            </button>
            <button
              type="button"
              onClick={handleClear}
              className="inline-flex min-h-11 flex-1 cursor-pointer items-center justify-center gap-2 rounded-md bg-raise px-4 py-2 text-sm font-semibold text-white transition-opacity duration-200 hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-raise focus-visible:ring-offset-2 focus-visible:ring-offset-bg sm:flex-none"
            >
              <Trash2 className="h-4 w-4" aria-hidden="true" />
              Clear
            </button>
          </div>
        </div>
      )}

      {records === null || stats === null || comparison === null ? (
        <LoadingState />
      ) : records.length === 0 ? (
        <EmptyState />
      ) : (
        <>
          <section aria-labelledby="summary-title">
            <div className="mb-4 flex items-center justify-between gap-3">
              <div>
                <p className="font-mono text-xs font-medium uppercase text-muted">
                  All practice
                </p>
                <h2 id="summary-title" className="mt-1 text-lg font-semibold text-fg">
                  Performance summary
                </h2>
              </div>
              <Link
                href="/practice"
                className="inline-flex min-h-11 cursor-pointer items-center justify-center gap-2 rounded-md border border-border px-3 py-2 text-sm font-medium text-fg transition-colors duration-200 hover:border-accent hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
              >
                <Zap className="h-4 w-4" aria-hidden="true" />
                <span className="hidden sm:inline">Practice again</span>
                <span className="sm:hidden">Practice</span>
              </Link>
            </div>
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              <SummaryCard
                icon={Crosshair}
                label="Decisions"
                value={stats.total.toLocaleString()}
                detail={`${stats.correct.toLocaleString()} correct`}
              />
              <SummaryCard
                icon={Target}
                label="Accuracy"
                value={formatPercent(stats.accuracy)}
                detail={`${stats.total - stats.correct} to review`}
              />
              <SummaryCard
                icon={Gauge}
                label="Average time"
                value={formatDuration(stats.averageResponseMs)}
                detail="Per decision"
              />
              <SummaryCard
                icon={Flame}
                label="Day streak"
                value={stats.streakDays.toLocaleString()}
                detail={
                  stats.streakDays === 1
                    ? 'Consecutive practice day'
                    : 'Consecutive practice days'
                }
              />
            </div>
          </section>

          <section
            aria-labelledby="trend-title"
            className="rounded-lg border border-border bg-surface p-4 sm:p-5"
          >
            <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
              <div className="flex items-start gap-3">
                <div className="grid h-9 w-9 shrink-0 place-items-center rounded-md bg-surface-2 text-accent">
                  {stats.trend < 0 ? (
                    <TrendingDown className="h-4 w-4" aria-hidden="true" />
                  ) : (
                    <TrendingUp className="h-4 w-4" aria-hidden="true" />
                  )}
                </div>
                <div>
                  <h2 id="trend-title" className="text-sm font-semibold text-fg">
                    Recent performance
                  </h2>
                  <p className="mt-0.5 text-xs leading-5 text-muted">
                    All range sets: your latest 20 decisions compared with the
                    20 before them.
                  </p>
                </div>
              </div>

              {comparison.previousCount === 20 ? (
                <div className="sm:text-right">
                  <p
                    className={
                      'font-mono text-xl font-semibold ' +
                      (stats.trend > 0
                        ? 'text-accent'
                        : stats.trend < 0
                          ? 'text-raise'
                          : 'text-fg')
                    }
                  >
                    {formatTrend(stats.trend)}
                  </p>
                  <p className="mt-0.5 text-xs text-muted">accuracy change</p>
                </div>
              ) : (
                <p className="max-w-xs text-xs leading-5 text-muted sm:text-right">
                  Record {Math.max(0, 40 - stats.total)} more{' '}
                  {40 - stats.total === 1 ? 'decision' : 'decisions'} to unlock a
                  comparison.
                </p>
              )}
            </div>

            <div className="mt-5 grid gap-4 sm:grid-cols-2">
              <div>
                <div className="flex items-center justify-between gap-3 text-xs">
                  <span className="font-medium text-fg">
                    Recent {comparison.recentCount}
                  </span>
                  <span className="font-mono font-semibold text-fg">
                    {formatPercent(comparison.recentAccuracy)}
                  </span>
                </div>
                <div className="mt-2 h-2 overflow-hidden rounded bg-surface-2">
                  <div
                    className="h-full rounded bg-accent"
                    style={{ width: `${comparison.recentAccuracy * 100}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between gap-3 text-xs">
                  <span className="font-medium text-fg">
                    Previous {comparison.previousCount}
                  </span>
                  <span className="font-mono font-semibold text-fg">
                    {comparison.previousCount > 0
                      ? formatPercent(comparison.previousAccuracy)
                      : 'Not available'}
                  </span>
                </div>
                <div className="mt-2 h-2 overflow-hidden rounded bg-surface-2">
                  {comparison.previousCount > 0 && (
                    <div
                      className="h-full rounded bg-muted"
                      style={{ width: `${comparison.previousAccuracy * 100}%` }}
                    />
                  )}
                </div>
              </div>
            </div>
          </section>

          <section aria-labelledby="breakdowns-title">
            <div className="mb-4">
              <p className="font-mono text-xs font-medium uppercase text-muted">
                Decision quality
              </p>
              <h2 id="breakdowns-title" className="mt-1 text-lg font-semibold text-fg">
                Strengths and focus areas
              </h2>
              <p className="mt-1 text-xs leading-5 text-muted">
                Each group is measured against your overall{' '}
                {formatPercent(stats.accuracy)} accuracy.
              </p>
            </div>
            <div className="grid items-start gap-4 lg:grid-cols-2 xl:grid-cols-3">
              <BreakdownPanel
                title="By table"
                description="Heads-up, 6-max, and full-ring results."
                icon={Table2}
                items={stats.byFormat}
                overallAccuracy={stats.accuracy}
              />
              <BreakdownPanel
                title="By position"
                description="Where you make the decision."
                icon={MapPin}
                items={stats.byPosition}
                overallAccuracy={stats.accuracy}
              />
              <BreakdownPanel
                title="By spot type"
                description="First-in decisions and responses."
                icon={Layers3}
                items={stats.byCategory}
                overallAccuracy={stats.accuracy}
              />
              <BreakdownPanel
                title="By best action"
                description="The chart's recommended response."
                icon={Check}
                items={stats.byAction}
                overallAccuracy={stats.accuracy}
              />
              <BreakdownPanel
                title="By range set"
                description="Stack depth and source used for each decision."
                icon={Gauge}
                items={stats.byScenario}
                overallAccuracy={stats.accuracy}
              />
            </div>
          </section>

          <section
            aria-labelledby="recent-title"
            className="overflow-hidden rounded-lg border border-border bg-surface"
          >
            <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-4 sm:px-5">
              <div>
                <h2 id="recent-title" className="text-sm font-semibold text-fg">
                  Recent decisions
                </h2>
                <p className="mt-0.5 text-xs text-muted">
                  Your last {stats.recent.length} answers
                </p>
              </div>
              <CalendarDays className="h-4 w-4 text-accent" aria-hidden="true" />
            </div>
            <div className="divide-y divide-border">
              {stats.recent.map((record) => (
                <article
                  key={record.id}
                  className="grid gap-3 px-4 py-4 sm:grid-cols-[52px_minmax(130px,1fr)_minmax(128px,0.8fr)_90px_116px] sm:items-center sm:px-5"
                >
                  <span className="w-fit rounded border border-border bg-surface-2 px-2 py-1 font-mono text-xs font-semibold text-fg">
                    {record.handClass}
                  </span>
                  <div>
                    <p className="text-sm font-medium text-fg">
                      {positionLabelForSeats(record.hero, record.seats)}
                      {record.villain
                        ? ` vs ${positionLabelForSeats(
                            record.villain,
                            record.seats
                          )}`
                        : ''}
                    </p>
                    <p className="mt-0.5 text-xs text-muted">
                      {record.category === 'RFI'
                        ? record.scenario?.openingSize.kind === 'all-in'
                          ? 'Push or fold'
                          : 'Raise first in'
                        : record.scenario?.openingSize.kind === 'all-in'
                          ? 'Facing a shove'
                          : 'Facing an open'}
                      {' · '}
                      {record.scenario?.label ?? 'Legacy curated baseline'}
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs sm:block">
                    <span className="text-muted">Played </span>
                    <span className="font-medium text-fg">{record.chosenAction}</span>
                    <span className="text-muted sm:block">
                      Default: {record.recommendedAction}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    {record.correct ? (
                      <Check className="h-4 w-4 text-accent" aria-hidden="true" />
                    ) : (
                      <X className="h-4 w-4 text-raise" aria-hidden="true" />
                    )}
                    <span
                      className={
                        'text-xs font-medium ' +
                        (record.correct ? 'text-accent' : 'text-raise')
                      }
                    >
                      {record.correct ? 'Correct' : 'Review'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between gap-3 text-xs text-muted sm:block sm:text-right">
                    <span className="inline-flex items-center gap-1">
                      <Clock3 className="h-3.5 w-3.5" aria-hidden="true" />
                      {formatDuration(record.responseMs)}
                    </span>
                    <time
                      dateTime={new Date(record.answeredAt).toISOString()}
                      className="sm:mt-1 sm:block"
                    >
                      {formatAnsweredAt(record.answeredAt)}
                    </time>
                  </div>
                </article>
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  );
}
