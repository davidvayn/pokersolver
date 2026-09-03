import {
  Activity,
  CalendarDays,
  Clock3,
  Gauge,
  Layers3,
  ShieldCheck,
  Target,
  TrendingDown,
  Zap,
} from 'lucide-react';
import { cardToStr } from '@/lib/cards';
import type {
  EvBreakdown,
  PracticeDecisionPoint,
  PracticeDecisionRecord,
  PracticeGrade,
  PracticeGradeBreakdown,
  PracticeHandRecord,
  PracticeStats,
  PracticeTrendPoint,
} from '@/lib/practice-types';

function bb(value: number | null, digits = 3): string {
  return value === null ? '—' : `${value.toFixed(digits)}bb`;
}

function duration(value: number | null): string {
  if (value === null) return '—';
  if (value === 0) return '0ms';
  return value < 1_000
    ? `${Math.round(value)}ms`
    : `${(value / 1_000).toFixed(1)}s`;
}

function percent(value: number | null): string {
  return value === null ? '—' : `${Math.round(value * 100)}%`;
}

function gradeColor(grade: PracticeGrade): string {
  switch (grade) {
    case 'perfect':
      return 'rgb(var(--accent))';
    case 'excellent':
      return 'rgb(var(--call))';
    case 'good':
      return 'rgb(var(--check))';
    case 'inaccuracy':
      return 'rgb(245 158 11)';
    case 'mistake':
      return 'rgb(249 115 22)';
    case 'blunder':
      return 'rgb(var(--raise))';
  }
}

function Metric({
  icon: Icon,
  label,
  value,
  detail,
}: {
  icon: typeof Gauge;
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="stats-overview-item min-w-0 p-4 sm:p-5">
      <div className="flex items-center gap-2 text-xs font-medium text-muted">
        <Icon className="h-4 w-4 text-accent" aria-hidden="true" />
        {label}
      </div>
      <p className="mt-3 font-mono text-2xl font-semibold tracking-tight sm:text-3xl">
        {value}
      </p>
      <p className="mt-1 truncate text-xs text-muted">{detail}</p>
    </div>
  );
}

function TrendChart({ points }: { points: PracticeTrendPoint[] }) {
  const width = 760;
  const height = 278;
  const left = 44;
  const right = 16;
  const top = 20;
  const bottom = 44;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const maxLoss = Math.max(
    0.05,
    ...points.map((point) => point.averageEvLossBb ?? 0)
  );
  const maxDecisions = Math.max(1, ...points.map((point) => point.decisions));
  const x = (index: number) =>
    left + (index / Math.max(1, points.length - 1)) * plotWidth;
  const y = (loss: number) => top + plotHeight - (loss / maxLoss) * plotHeight;
  const plotted = points
    .map((point, index) => ({ point, index }))
    .filter(({ point }) => point.averageEvLossBb !== null);
  const path = plotted
    .map(({ point, index }, plottedIndex) => `${plottedIndex ? 'L' : 'M'} ${x(index).toFixed(1)} ${y(point.averageEvLossBb ?? 0).toFixed(1)}`)
    .join(' ');

  return (
    <figure>
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold">Learning curve</h2>
          <p className="mt-1 text-xs text-muted">Daily EV loss and decision volume · 21 days</p>
        </div>
        <div className="flex items-center gap-4 text-[11px] text-muted" aria-hidden="true">
          <span className="flex items-center gap-1.5"><span className="h-0.5 w-4 bg-accent" /> EV loss</span>
          <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm bg-accent/15" /> Decisions</span>
        </div>
      </div>
      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="h-auto w-full min-w-[560px]"
          role="img"
          aria-label={`${plotted.length} days with graded decisions. Lower EV loss is better.`}
        >
          {[0, 0.5, 1].map((ratio) => {
            const lineY = top + plotHeight * ratio;
            const label = maxLoss * (1 - ratio);
            return (
              <g key={ratio}>
                <line x1={left} x2={width - right} y1={lineY} y2={lineY} stroke="rgb(var(--border))" strokeDasharray="3 5" />
                <text x={left - 8} y={lineY + 4} textAnchor="end" className="fill-muted text-[10px]">{label.toFixed(2)}</text>
              </g>
            );
          })}
          {points.map((point, index) => {
            const barHeight = (point.decisions / maxDecisions) * plotHeight;
            return (
              <rect key={point.key} x={x(index) - 6} y={top + plotHeight - barHeight} width="12" height={barHeight} rx="2" fill="rgb(var(--accent) / 0.12)">
                <title>{`${point.label}: ${point.decisions} decisions${point.averageEvLossBb === null ? '' : `, ${bb(point.averageEvLossBb)} average EV loss`}`}</title>
              </rect>
            );
          })}
          {path && <path d={path} fill="none" stroke="rgb(var(--accent))" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" pathLength="1" className="stats-chart-line" />}
          {points.map((point, index) => point.averageEvLossBb === null ? null : (
            <circle key={point.key} cx={x(index)} cy={y(point.averageEvLossBb)} r="3.5" fill="rgb(var(--surface))" stroke="rgb(var(--accent))" strokeWidth="2">
              <title>{`${point.label}: ${bb(point.averageEvLossBb)} average EV loss, ${point.decisions} decisions`}</title>
            </circle>
          ))}
          {points.map((point, index) => index % 5 === 0 || index === points.length - 1 ? (
            <text key={point.key} x={x(index)} y={height - 14} textAnchor={index === 0 ? 'start' : index === points.length - 1 ? 'end' : 'middle'} className="fill-muted text-[10px]">{point.label}</text>
          ) : null)}
        </svg>
      </div>
      {plotted.length === 0 && <figcaption className="mt-2 text-xs text-muted">No graded decisions in this window.</figcaption>}
      <details className="mt-2 text-xs">
        <summary className="flex min-h-11 cursor-pointer items-center text-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent">
          View daily values
        </summary>
        <div className="overflow-x-auto border-t border-border pt-2">
          <table className="w-full min-w-[420px] text-left">
            <thead className="text-[10px] uppercase text-muted">
              <tr><th className="py-2 font-medium">Day</th><th className="py-2 text-right font-medium">Decisions</th><th className="py-2 text-right font-medium">Strong</th><th className="py-2 text-right font-medium">Avg loss</th></tr>
            </thead>
            <tbody className="divide-y divide-border">
              {points.filter((point) => point.decisions > 0).map((point) => (
                <tr key={point.key}>
                  <td className="py-2">{point.label}</td>
                  <td className="py-2 text-right font-mono">{point.decisions}</td>
                  <td className="py-2 text-right font-mono">{percent(point.strongDecisionPercentage)}</td>
                  <td className="py-2 text-right font-mono">{bb(point.averageEvLossBb)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}

function GradeDonut({ distribution, strongPercentage }: { distribution: PracticeGradeBreakdown[]; strongPercentage: number }) {
  const circumference = 2 * Math.PI * 46;
  return (
    <div>
      <h2 className="text-sm font-semibold">Decision quality</h2>
      <p className="mt-1 text-xs text-muted">Frequency grades · all time</p>
      <div className="mt-5 grid grid-cols-[132px_minmax(0,1fr)] items-center gap-5">
        <div className="relative">
          <svg viewBox="0 0 120 120" className="stats-quality-ring h-32 w-32 -rotate-90" role="img" aria-label={`${percent(strongPercentage)} of decisions were perfect, excellent, or good`}>
            <circle cx="60" cy="60" r="46" fill="none" stroke="rgb(var(--surface-2))" strokeWidth="13" />
            <circle cx="60" cy="60" r="46" fill="none" stroke="rgb(var(--accent))" strokeWidth="13" strokeLinecap="round" strokeDasharray={`${strongPercentage * circumference} ${circumference}`} />
          </svg>
          <div className="absolute inset-0 grid place-content-center text-center">
            <strong className="font-mono text-2xl">{percent(strongPercentage)}</strong>
            <span className="text-[10px] text-muted">strong</span>
          </div>
        </div>
        <div className="space-y-2">
          {distribution.map((item) => (
            <div key={item.grade} className="flex items-center gap-2 text-[11px]">
              <span className="h-2 w-2 rounded-full" style={{ background: gradeColor(item.grade) }} />
              <span className="min-w-0 flex-1 truncate">{item.label}</span>
              <span className="font-mono text-muted">{item.decisions}</span>
              <span className="w-8 text-right font-mono text-muted">{percent(item.percentage)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ActivityHeatmap({ stats }: { stats: PracticeStats }) {
  const max = Math.max(1, ...stats.activity.map((day) => day.decisions));
  const total = stats.activity.reduce((sum, day) => sum + day.decisions, 0);
  return (
    <figure className="min-w-0 rounded-lg border border-border bg-surface p-4 sm:p-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold">Practice rhythm</h2>
          <p className="mt-1 text-xs text-muted">{total} decisions across the last 12 weeks</p>
        </div>
        <div className="flex items-center gap-3 font-mono text-[11px] text-muted"><span>{stats.activeDays} active days</span><span>{stats.currentStreakDays} day streak</span></div>
      </div>
      <div className="mt-5 grid grid-flow-col grid-rows-7 gap-1.5" role="img" aria-label={`Practice activity over 12 weeks. ${total} decisions, ${stats.activeDays} active days all time, current streak ${stats.currentStreakDays} days.`}>
        {stats.activity.map((day) => {
          const intensity = day.decisions === 0 ? 0 : 0.2 + (day.decisions / max) * 0.8;
          return <span key={day.key} className="aspect-square min-w-0 rounded-[3px] border border-border/60" style={{ backgroundColor: day.decisions === 0 ? 'rgb(var(--surface-2))' : `rgb(var(--accent) / ${intensity.toFixed(2)})` }} title={`${day.label}: ${day.decisions} decision${day.decisions === 1 ? '' : 's'}`} aria-hidden="true" />;
        })}
      </div>
      <figcaption className="mt-3 flex items-center justify-end gap-1.5 text-[10px] text-muted">
        Less
        {[0, 0.22, 0.45, 0.7, 1].map((opacity) => <span key={opacity} className="h-2.5 w-2.5 rounded-[2px] border border-border/60" style={{ backgroundColor: opacity ? `rgb(var(--accent) / ${opacity})` : 'rgb(var(--surface-2))' }} />)}
        More
      </figcaption>
    </figure>
  );
}

function BreakdownChart({ title, subtitle, items }: { title: string; subtitle: string; items: EvBreakdown[] }) {
  const visible = items.slice(0, 8);
  const maxLoss = Math.max(0.01, ...visible.map((item) => item.averageEvLossBb ?? 0));
  return (
    <section className="rounded-lg border border-border bg-surface p-4 sm:p-5">
      <h3 className="text-sm font-semibold">{title}</h3>
      <p className="mt-1 text-xs text-muted">{subtitle}</p>
      {visible.length === 0 ? <p className="mt-8 text-sm text-muted">No decisions yet.</p> : (
        <div className="mt-5 space-y-4">
          {visible.map((item) => {
            const width = item.averageEvLossBb === null ? 0 : Math.max(2, (item.averageEvLossBb / maxLoss) * 100);
            return (
              <div key={item.key}>
                <div className="flex items-baseline justify-between gap-3 text-xs"><span className="min-w-0 truncate font-medium capitalize">{item.label}</span><span className="shrink-0 font-mono text-muted">{bb(item.averageEvLossBb)} · {item.decisions}</span></div>
                <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-surface-2"><span className="stats-bar-fill block h-full origin-left rounded-full bg-raise" style={{ width: `${width}%`, opacity: item.averageEvLossBb === null ? 0 : 0.78 }} /></div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

function percentile(values: number[], ratio: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((first, second) => first - second);
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * ratio))];
}

function ScatterPlot({ points }: { points: PracticeDecisionPoint[] }) {
  const width = 650;
  const height = 300;
  const left = 52;
  const right = 18;
  const top = 18;
  const bottom = 48;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const maxResponse = Math.max(1_000, percentile(points.map((point) => point.responseMs), 0.95));
  const maxLoss = Math.max(0.05, percentile(points.map((point) => point.evLossBb), 0.95));
  const medianResponse = percentile(points.map((point) => point.responseMs), 0.5);
  const medianLoss = percentile(points.map((point) => point.evLossBb), 0.5);
  const x = (response: number) => left + (Math.min(response, maxResponse) / maxResponse) * plotWidth;
  const y = (loss: number) => top + plotHeight - (Math.min(loss, maxLoss) / maxLoss) * plotHeight;
  return (
    <figure className="min-w-0 rounded-lg border border-border bg-surface p-4 sm:p-5">
      <h2 className="text-sm font-semibold">Speed × precision</h2>
      <p className="mt-1 text-xs text-muted">Last {points.length} graded decisions · lower-left is faster and cheaper · axes cap outliers</p>
      {points.length < 2 ? <p className="mt-10 text-sm text-muted">Two graded decisions unlock this view.</p> : (
        <div className="mt-3 overflow-x-auto">
          <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full min-w-[500px]" role="img" aria-label={`Scatter plot of ${points.length} decisions by response time and EV loss`}>
            {[0, 0.5, 1].map((ratio) => (
              <g key={ratio}>
                <line x1={left} x2={width - right} y1={top + plotHeight * ratio} y2={top + plotHeight * ratio} stroke="rgb(var(--border))" strokeDasharray="3 5" />
                <text x={left - 8} y={top + plotHeight * ratio + 4} textAnchor="end" className="fill-muted text-[10px]">{(maxLoss * (1 - ratio)).toFixed(2)}</text>
                <text x={left + plotWidth * ratio} y={height - 17} textAnchor={ratio === 0 ? 'start' : ratio === 1 ? 'end' : 'middle'} className="fill-muted text-[10px]">{duration(maxResponse * ratio)}</text>
              </g>
            ))}
            <line x1={x(medianResponse)} x2={x(medianResponse)} y1={top} y2={top + plotHeight} stroke="rgb(var(--muted) / 0.45)" strokeDasharray="5 5" />
            <line x1={left} x2={width - right} y1={y(medianLoss)} y2={y(medianLoss)} stroke="rgb(var(--muted) / 0.45)" strokeDasharray="5 5" />
            {points.map((point) => <circle key={point.id} cx={x(point.responseMs)} cy={y(point.evLossBb)} r="5" fill={gradeColor(point.grade)} fillOpacity="0.72" stroke="rgb(var(--surface))" strokeWidth="1.5"><title>{`${point.label}: ${duration(point.responseMs)}, ${bb(point.evLossBb)} EV loss`}</title></circle>)}
            <text x={left + plotWidth / 2} y={height - 2} textAnchor="middle" className="fill-muted text-[10px]">Response time</text>
            <text x="11" y={top + plotHeight / 2} textAnchor="middle" transform={`rotate(-90 11 ${top + plotHeight / 2})`} className="fill-muted text-[10px]">EV loss (bb)</text>
          </svg>
        </div>
      )}
    </figure>
  );
}

function ProgressMetric({ label, value, detail }: { label: string; value: number; detail: string }) {
  return (
    <div>
      <div className="flex items-baseline justify-between gap-4 text-xs"><span className="font-medium">{label}</span><span className="font-mono font-semibold">{percent(value)}</span></div>
      <div className="mt-2 h-2 overflow-hidden rounded-full bg-surface-2"><span className="stats-bar-fill block h-full origin-left rounded-full bg-accent" style={{ width: `${value * 100}%` }} /></div>
      <p className="mt-1.5 text-[11px] text-muted">{detail}</p>
    </div>
  );
}

function TrainingHealth({ stats }: { stats: PracticeStats }) {
  return (
    <section className="rounded-lg border border-border bg-surface p-4 sm:p-5">
      <h2 className="text-sm font-semibold">Training health</h2>
      <p className="mt-1 text-xs text-muted">Coverage, confidence, and consistency</p>
      <div className="mt-5 space-y-5">
        <ProgressMetric label="Strong decisions" value={stats.strongDecisionPercentage} detail={`${stats.strongDecisions} perfect, excellent, or good grades`} />
        <ProgressMetric label="EV coverage" value={stats.gradedCoveragePercentage} detail={`${stats.gradedDecisions} decisions include comparable action EVs`} />
        <ProgressMetric label="High confidence" value={1 - stats.lowConfidencePercentage} detail="Action-EV estimates without a confidence warning" />
      </div>
      <dl className="mt-6 grid grid-cols-2 gap-x-5 gap-y-4 border-t border-border pt-5">
        {[
          ['Total EV loss', stats.gradedDecisions ? bb(stats.totalEvLossBb) : '—'],
          ['Recent EV shift', stats.trendEvLossBb === null ? '—' : `${stats.trendEvLossBb > 0 ? '+' : ''}${bb(stats.trendEvLossBb)}`],
          ['Longest streak', `${stats.longestStreakDays}d`],
          ['Active days', String(stats.activeDays)],
          ['Decisions / hand', stats.decisionsPerHand.toFixed(1)],
          ['Average hand', duration(stats.averageHandDurationMs)],
        ].map(([label, value]) => <div key={label}><dt className="text-[11px] text-muted">{label}</dt><dd className="mt-1 font-mono text-lg font-semibold">{value}</dd></div>)}
      </dl>
      <p className="mt-4 text-[11px] leading-5 text-muted">Recent EV shift compares the latest 50 decisions with the previous 50. A negative result is improvement.</p>
    </section>
  );
}

function CostlyDecision({ decision, hand }: { decision: PracticeDecisionRecord; hand?: PracticeHandRecord }) {
  return (
    <details className="group border-b border-border last:border-b-0">
      <summary className="flex min-h-16 cursor-pointer list-none items-center justify-between gap-4 px-4 py-3 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent sm:px-5">
        <div className="min-w-0"><p className="truncate text-sm font-semibold">{decision.handBucket} · {decision.street} · {decision.chosenAction.label}</p><p className="mt-1 font-mono text-xs text-muted">{decision.heroCards.map(cardToStr).join(' ')} · {decision.depthBb}bb · {decision.position === 'button-small-blind' ? 'BTN / SB' : 'BB'}</p></div>
        <div className="shrink-0 text-right"><p className="font-mono text-sm font-semibold text-raise">{bb(decision.evLossBb)}</p><p className="mt-1 text-[11px] capitalize text-muted">{decision.grade}</p></div>
      </summary>
      <div className="border-t border-border bg-surface-2/45 px-4 py-4 text-xs sm:px-5">
        <div className="grid gap-4 sm:grid-cols-2">
          <div><p className="font-semibold">Complete line</p><ol className="mt-2 space-y-1 text-muted">{hand?.actions.map((action) => <li key={action.id}>{action.actor === hand.hero ? 'Hero' : 'Opponent'} · {action.street} · {action.label}</li>) ?? <li>Action history unavailable</li>}</ol>{hand && <p className="mt-3 font-mono text-muted">Board {hand.board.length ? hand.board.map(cardToStr).join(' ') : '—'} · Opponent {hand.opponentCards.map(cardToStr).join(' ')}</p>}</div>
          <div><p className="font-semibold">Policy mix</p><div className="mt-2 space-y-2">{decision.policyActions.length ? decision.policyActions.map((action) => <div key={action.id} className="flex justify-between gap-3 text-muted"><span>{action.label}</span><span className="font-mono">{percent(action.probability)} · {action.evBb === null ? 'EV unavailable' : `${action.evBb.toFixed(3)}bb`}</span></div>) : <p className="text-muted">Policy mix unavailable.</p>}</div></div>
        </div>
        {decision.lowConfidence && <p className="mt-4 border-l-2 border-amber-500 bg-amber-500/10 p-3 leading-5">This record carries a {decision.confidence} confidence warning for its EV estimate.</p>}
      </div>
    </details>
  );
}

export function PracticeStatsDashboard({ stats, hands }: { stats: PracticeStats; hands: PracticeHandRecord[] }) {
  const handById = new Map(hands.map((hand) => [hand.id, hand]));
  return (
    <div className="stats-dashboard mt-6 space-y-5">
      <section className="stats-overview grid overflow-hidden rounded-lg border border-border bg-surface sm:grid-cols-2 xl:grid-cols-4" aria-label="Practice overview">
        <Metric icon={Layers3} label="Decisions" value={String(stats.decisions)} detail={`${stats.hands} complete hands`} />
        <Metric icon={Target} label="Strong decisions" value={percent(stats.strongDecisionPercentage)} detail={`${stats.strongDecisions} strong grades`} />
        <Metric icon={TrendingDown} label="Average EV loss" value={bb(stats.averageEvLossBb)} detail={`${stats.gradedDecisions} graded decisions`} />
        <Metric icon={Zap} label="Average response" value={duration(stats.averageResponseMs)} detail={`${stats.currentStreakDays} day current streak`} />
      </section>
      <div className="grid min-w-0 gap-5 xl:grid-cols-12">
        <section className="min-w-0 rounded-lg border border-border bg-surface p-4 sm:p-5 xl:col-span-8"><TrendChart points={stats.dailyTrend} /></section>
        <section className="min-w-0 rounded-lg border border-border bg-surface p-4 sm:p-5 xl:col-span-4"><GradeDonut distribution={stats.gradeDistribution} strongPercentage={stats.strongDecisionPercentage} /></section>
      </div>
      <ActivityHeatmap stats={stats} />
      <div className="grid min-w-0 gap-5 xl:grid-cols-12"><div className="min-w-0 xl:col-span-7"><ScatterPlot points={stats.decisionPoints} /></div><div className="min-w-0 xl:col-span-5"><TrainingHealth stats={stats} /></div></div>
      <section aria-labelledby="performance-breakdowns">
        <div className="mb-4 flex items-center gap-2"><Activity className="h-4 w-4 text-accent" aria-hidden="true" /><div><h2 id="performance-breakdowns" className="text-sm font-semibold">Performance map</h2><p className="mt-0.5 text-xs text-muted">Average EV loss · decision count</p></div></div>
        <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
          <BreakdownChart title="By street" subtitle="Where errors enter the hand" items={stats.byStreet} />
          <BreakdownChart title="By position" subtitle="BTN / SB versus big blind" items={stats.byPosition} />
          <BreakdownChart title="By stack" subtitle="Performance by effective depth" items={stats.byStack} />
          <BreakdownChart title="By action" subtitle="Cost of the actions you choose" items={stats.byAction} />
          <BreakdownChart title="By mode" subtitle="Results across practice formats" items={stats.byMode} />
          <BreakdownChart title="By response time" subtitle="Speed bands versus EV loss" items={stats.byResponseTime} />
        </div>
      </section>
      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_380px]">
        <section className="overflow-hidden rounded-lg border border-border bg-surface">
          <div className="border-b border-border px-4 py-4 sm:px-5"><div className="flex items-center gap-2"><Gauge className="h-4 w-4 text-accent" aria-hidden="true" /><h2 className="text-sm font-semibold">Review queue</h2></div><p className="mt-1 text-xs text-muted">Highest EV-loss decisions · expand for the hand and policy</p></div>
          {stats.recentCostly.length ? stats.recentCostly.slice(0, 8).map((decision) => <CostlyDecision key={decision.id} decision={decision} hand={handById.get(decision.handId)} />) : <p className="p-5 text-sm text-muted">No graded costly decisions yet.</p>}
        </section>
        <section className="rounded-lg border border-border bg-surface p-4 sm:p-5">
          <div className="flex items-center gap-2"><ShieldCheck className="h-4 w-4 text-accent" aria-hidden="true" /><h2 className="text-sm font-semibold">Focus next</h2></div><p className="mt-1 text-xs leading-5 text-muted">Repeated groups with the highest average EV loss.</p>
          <ol className="mt-5 space-y-5">{stats.weaknesses.length ? stats.weaknesses.map((item, index) => <li key={item.key} className="grid grid-cols-[24px_minmax(0,1fr)] gap-3"><span className="grid h-6 w-6 place-content-center rounded-full bg-raise/10 font-mono text-[10px] font-semibold text-raise">{index + 1}</span><div className="min-w-0"><p className="text-xs font-semibold capitalize leading-5">{item.label}</p><p className="mt-1 font-mono text-[11px] text-muted">{bb(item.averageEvLossBb)} avg · {item.decisions} attempts</p></div></li>) : <li className="text-xs text-muted">Two attempts in the same spot unlock a focus recommendation.</li>}</ol>
          <div className="mt-6 border-t border-border pt-4 text-[11px] leading-5 text-muted"><div className="flex items-start gap-2"><CalendarDays className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />The Practice adaptive sampler uses the latest 200 graded decisions while preserving 30% authentic random coverage.</div><div className="mt-2 flex items-start gap-2"><Clock3 className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />Response-time bands are descriptive; grading still comes only from the frozen policy mix.</div></div>
        </section>
      </div>
    </div>
  );
}
