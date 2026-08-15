import {
  AlertTriangle,
  BarChart3,
  Check,
  ClipboardList,
  Info,
  Settings2,
  Sparkles,
} from 'lucide-react';
import { cardToStr } from '@/lib/cards';
import type {
  OpponentModelSnapshot,
  PolicyManifest,
  PracticeDecisionRecord,
  PracticeHandRecord,
  PracticeSettings,
} from '@/lib/practice-types';
import { pushFoldDepths } from '@/lib/push-fold-policy';

export type RailTab = 'feedback' | 'history' | 'settings' | 'stats';

interface AnalystRailProps {
  idPrefix?: string;
  tab: RailTab;
  onTabChange: (tab: RailTab) => void;
  feedback: PracticeDecisionRecord | null;
  recentHands: PracticeHandRecord[];
  settings: PracticeSettings;
  pendingSettings: PracticeSettings | null;
  onSettingsChange: (settings: PracticeSettings) => void;
  fullDepths: number[];
  manifest: PolicyManifest | null;
  sessionDecisions: PracticeDecisionRecord[];
  historyWarning: string;
  opponentModel: OpponentModelSnapshot | null;
}

const TABS: Array<{ value: RailTab; label: string; icon: typeof Info }> = [
  { value: 'feedback', label: 'Feedback', icon: Sparkles },
  { value: 'history', label: 'History', icon: ClipboardList },
  { value: 'settings', label: 'Settings', icon: Settings2 },
  { value: 'stats', label: 'Run stats', icon: BarChart3 },
];

function pct(value: number): string {
  return `${(value * 100).toFixed(value > 0.995 ? 1 : 0)}%`;
}

function estimatedActionLoss(
  bestActionEvBb: number | null,
  actionEvBb: number | null
): number | null {
  if (bestActionEvBb === null || actionEvBb === null) return null;
  return Math.max(0, bestActionEvBb - actionEvBb);
}

function FeedbackPanel({ feedback }: { feedback: PracticeDecisionRecord | null }) {
  if (!feedback) {
    return (
      <div className="px-4 py-10 text-center">
        <Sparkles className="mx-auto h-5 w-5 text-accent" aria-hidden="true" />
        <p className="mt-3 text-sm font-medium">Your review appears here</p>
        <p className="mt-1 text-xs leading-5 text-muted">
          Choose an action to see the complete policy mix, action values, and confidence.
        </p>
      </div>
    );
  }
  return (
    <div className="space-y-5 p-4">
      <div>
        <div className="flex items-center justify-between gap-3">
          <span className="text-xs font-semibold uppercase text-muted">Decision</span>
          <span className="rounded-full bg-surface-2 px-2 py-1 text-[11px] font-semibold capitalize">
            {feedback.grade}
          </span>
        </div>
        <p className="mt-2 text-lg font-semibold">{feedback.chosenAction.label}</p>
        <p className="mt-1 font-mono text-xs text-muted">
          {feedback.handBucket} · {feedback.position === 'button-small-blind' ? 'BTN / SB' : 'BB'}
        </p>
      </div>

      <div className="space-y-3">
        {feedback.policyActions.map((action) => {
          const loss = estimatedActionLoss(
            feedback.bestActionEvBb,
            action.evBb
          );
          const isBestEstimate = loss !== null && loss <= 1e-9;
          return (
            <div key={action.id}>
              <div className="flex items-center justify-between gap-2 text-xs">
                <span className="flex flex-wrap items-center gap-1.5 font-medium">
                  {action.id === feedback.chosenAction.id && (
                    <Check className="h-3.5 w-3.5 text-accent" aria-hidden="true" />
                  )}
                  {action.label}
                  {isBestEstimate && (
                    <span className="rounded-full bg-accent/10 px-1.5 py-0.5 text-[10px] font-semibold text-accent">
                      Best estimated EV
                    </span>
                  )}
                </span>
                <span className="font-mono">{pct(action.probability)}</span>
              </div>
              <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-surface-2">
                <div
                  className="h-full rounded-full bg-accent"
                  style={{ width: `${Math.max(1, action.probability * 100)}%` }}
                />
              </div>
              <p className="mt-1 text-[11px] text-muted">
                {action.evBb === null
                  ? 'Action EV and estimated loss unavailable in this model version'
                  : `${action.evBb.toFixed(3)}bb EV${
                      action.standardErrorBb === null
                        ? ' · uncertainty unavailable'
                        : ` ± ${action.standardErrorBb.toFixed(3)}bb`
                    } · Estimated loss ${loss?.toFixed(3) ?? '—'}bb`}
              </p>
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-2 gap-2 border-t border-border pt-4 text-xs">
        <div>
          <p className="text-muted">Estimated EV loss</p>
          <p className="mt-1 font-mono font-semibold">
            {feedback.evLossBb === null ? 'Not graded' : `${feedback.evLossBb.toFixed(3)}bb`}
          </p>
        </div>
        <div>
          <p className="text-muted">Response</p>
          <p className="mt-1 font-mono font-semibold">{(feedback.responseMs / 1000).toFixed(1)}s</p>
        </div>
      </div>

      {feedback.lowConfidence && (
        <div className="flex gap-2 rounded-md border border-amber-500/35 bg-amber-500/10 p-3 text-xs leading-5">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600 dark:text-amber-300" aria-hidden="true" />
          <p>
            {feedback.confidence === 'unavailable'
              ? 'This model version does not contain per-action EV estimates. The choice is saved as ungraded.'
              : 'This action value has a wider sampling or low-reach uncertainty bound. Treat the EV-loss grade as approximate.'}
          </p>
        </div>
      )}

      {feedback.opponentModel && (
        <div className="rounded-md border border-border bg-surface-2 p-3 text-xs leading-5">
          <p className="font-semibold">Opponent adaptation for this hand</p>
          <p className="mt-1 text-muted">
            {feedback.opponentModel.observations} local observations · response weight{' '}
            {pct(feedback.opponentModel.responseWeight)} · confidence{' '}
            {pct(feedback.opponentModel.confidence)}
          </p>
          <p className="mt-1 text-muted">
            Your decision was graded against the frozen baseline.
          </p>
        </div>
      )}
    </div>
  );
}

function SettingsPanel({
  settings,
  pendingSettings,
  onSettingsChange,
  fullDepths,
}: Pick<
  AnalystRailProps,
  'settings' | 'pendingSettings' | 'onSettingsChange' | 'fullDepths'
>) {
  const shown = pendingSettings ?? settings;
  const patch = (next: Partial<PracticeSettings>) =>
    onSettingsChange({ ...shown, ...next });
  return (
    <div className="space-y-5 p-4">
      {pendingSettings && (
        <div className="rounded-md border border-accent/35 bg-accent/10 p-3 text-xs leading-5">
          <p className="font-semibold">Settings queued</p>
          <p className="mt-1 text-muted">Structural changes apply after the current hand.</p>
        </div>
      )}

      <fieldset>
        <legend className="text-xs font-semibold uppercase text-muted">Mode</legend>
        <div className="mt-2 grid grid-cols-2 gap-2">
          {(
            [
              ['full-hand', 'Full hand'],
              ['preflop', 'Preflop'],
              ['postflop', 'Postflop'],
              ['push-fold', 'Push/fold'],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              type="button"
              aria-pressed={shown.mode === value}
              onClick={() =>
                patch(value === 'push-fold' ? { mode: value, dealMode: 'authentic' } : { mode: value })
              }
              className={`min-h-11 rounded-md border px-3 text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${shown.mode === value ? 'border-accent bg-accent text-accent-fg' : 'border-border hover:border-accent/60'}`}
            >
              {label}
            </button>
          ))}
        </div>
      </fieldset>

      {shown.mode === 'push-fold' ? (
        <label className="block text-xs font-semibold uppercase text-muted">
          Effective stack
          <select
            value={shown.pushFoldDepthBb}
            onChange={(event) =>
              patch({ pushFoldDepthBb: Number(event.target.value) as PracticeSettings['pushFoldDepthBb'] })
            }
            className="mt-2 min-h-11 w-full rounded-md border border-border bg-bg px-3 text-sm font-medium text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            {pushFoldDepths().map((depth) => (
              <option key={depth} value={depth}>{depth}bb</option>
            ))}
          </select>
        </label>
      ) : (
        <div>
          <label className="block text-xs font-semibold uppercase text-muted">
            Effective stack
            <select
              value={shown.depthBb}
              disabled={fullDepths.length === 0}
              onChange={(event) =>
                patch({ depthBb: Number(event.target.value) as PracticeSettings['depthBb'] })
              }
              className="mt-2 min-h-11 w-full rounded-md border border-border bg-bg px-3 text-sm font-medium text-fg disabled:cursor-not-allowed disabled:opacity-60"
            >
              {fullDepths.length > 0 ? (
                fullDepths.map((depth) => <option key={depth} value={depth}>{depth}bb</option>)
              ) : (
                <option value={20}>No validated depth</option>
              )}
            </select>
          </label>
          {fullDepths.length === 0 && (
            <p className="mt-2 text-xs leading-5 text-muted">
              20/50/100bb stay hidden until their independent two-seed validation passes.
            </p>
          )}
        </div>
      )}

      {shown.mode === 'postflop' && (
        <fieldset>
          <legend className="text-xs font-semibold uppercase text-muted">Streets</legend>
          <div className="mt-2 flex flex-wrap gap-2">
            {(['flop', 'turn', 'river'] as const).map((street) => {
              const selected = shown.postflopStreets.includes(street);
              return (
                <button
                  key={street}
                  type="button"
                  aria-pressed={selected}
                  onClick={() => {
                    const next = selected
                      ? shown.postflopStreets.filter((item) => item !== street)
                      : [...shown.postflopStreets, street];
                    if (next.length > 0) patch({ postflopStreets: next });
                  }}
                  className={`min-h-11 rounded-md border px-3 text-sm capitalize ${selected ? 'border-accent bg-accent/10' : 'border-border'}`}
                >
                  {street}
                </button>
              );
            })}
          </div>
        </fieldset>
      )}

      <label className="block text-xs font-semibold uppercase text-muted">
        Hero seat
        <select
          value={shown.heroSeat}
          onChange={(event) => patch({ heroSeat: event.target.value as PracticeSettings['heroSeat'] })}
          className="mt-2 min-h-11 w-full rounded-md border border-border bg-bg px-3 text-sm font-medium text-fg"
        >
          <option value="alternate">Alternate</option>
          <option value="button-small-blind">BTN / SB only</option>
          <option value="big-blind">BB only</option>
        </select>
      </label>

      <label className="block text-xs font-semibold uppercase text-muted">
        Deal mode
        <select
          value={shown.dealMode}
          onChange={(event) => patch({ dealMode: event.target.value as PracticeSettings['dealMode'] })}
          className="mt-2 min-h-11 w-full rounded-md border border-border bg-bg px-3 text-sm font-medium text-fg"
        >
          <option value="authentic">Authentic random</option>
          <option value="adaptive" disabled>Adaptive (70/30)</option>
        </select>
        <span className="mt-2 block text-xs font-normal normal-case leading-5 text-muted">
          Adaptive dealing activates after a validated graded sample corpus is installed.
        </span>
      </label>

      {shown.mode !== 'push-fold' && (
        <label className="block text-xs font-semibold uppercase text-muted">
          Opponent policy
          <select
            value={shown.opponentStyle}
            onChange={(event) =>
              patch({
                opponentStyle: event.target
                  .value as PracticeSettings['opponentStyle'],
              })
            }
            className="mt-2 min-h-11 w-full rounded-md border border-border bg-bg px-3 text-sm font-medium text-fg"
          >
            <option value="adaptive-exploitative">Adaptive exploitative</option>
            <option value="baseline">Frozen baseline</option>
          </select>
          <span className="mt-2 block text-xs font-normal normal-case leading-5 text-muted">
            Adaptation uses only local history, starts at zero, and remains capped by the pinned model.
          </span>
        </label>
      )}

      <label className="block text-xs font-semibold uppercase text-muted">
        Optional goal
        <select
          value={shown.decisionGoal}
          onChange={(event) =>
            patch({
              decisionGoal:
                event.target.value === 'continuous'
                  ? 'continuous'
                  : (Number(event.target.value) as 25 | 50 | 100),
            })
          }
          className="mt-2 min-h-11 w-full rounded-md border border-border bg-bg px-3 text-sm font-medium text-fg"
        >
          <option value="continuous">Continuous</option>
          <option value={25}>25 decisions</option>
          <option value={50}>50 decisions</option>
          <option value={100}>100 decisions</option>
        </select>
      </label>
    </div>
  );
}

function HistoryPanel({ recentHands }: { recentHands: PracticeHandRecord[] }) {
  if (recentHands.length === 0) {
    return <p className="px-4 py-10 text-center text-sm text-muted">No hands in the new practice history yet.</p>;
  }
  return (
    <ol className="divide-y divide-border">
      {recentHands.slice(0, 20).map((hand) => (
        <li key={hand.id} className="p-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-sm font-semibold capitalize">{hand.mode.replace('-', ' ')}</p>
              <p className="mt-1 font-mono text-xs text-muted">
                {hand.heroCards.map(cardToStr).join(' ')} · {hand.depthBb}bb
              </p>
            </div>
            <span className="text-xs text-muted">{hand.decisions.length} decision{hand.decisions.length === 1 ? '' : 's'}</span>
          </div>
          <p className="mt-2 text-xs text-muted">
            {hand.result.winner === 'split'
              ? 'Split pot'
              : hand.result.winner === hand.hero
                ? `Won ${hand.result.potBb.toFixed(1)}bb pot`
                : hand.result.winner
                  ? `Lost ${hand.result.potBb.toFixed(1)}bb pot`
                  : 'Round complete'}
          </p>
          {hand.opponentModel && (
            <p className="mt-1 text-xs text-muted">
              Opponent response {pct(hand.opponentModel.responseWeight)} from{' '}
              {hand.opponentModel.observations} local observations
            </p>
          )}
        </li>
      ))}
    </ol>
  );
}

function StatsPanel({ decisions }: { decisions: PracticeDecisionRecord[] }) {
  const graded = decisions.filter((decision) => decision.evLossBb !== null);
  const total = graded.reduce((sum, decision) => sum + (decision.evLossBb ?? 0), 0);
  const low = decisions.filter((decision) => decision.lowConfidence).length;
  return (
    <div className="p-4">
      <div className="grid grid-cols-2 gap-px overflow-hidden rounded-md border border-border bg-border">
        {[
          ['Decisions', String(decisions.length)],
          ['Graded', String(graded.length)],
          ['Avg EV loss', graded.length ? `${(total / graded.length).toFixed(3)}bb` : '—'],
          ['Low confidence', decisions.length ? `${Math.round((low / decisions.length) * 100)}%` : '—'],
        ].map(([label, value]) => (
          <div key={label} className="bg-surface p-3">
            <p className="text-[11px] text-muted">{label}</p>
            <p className="mt-1 font-mono text-base font-semibold">{value}</p>
          </div>
        ))}
      </div>
      <p className="mt-4 text-xs leading-5 text-muted">
        Goals finish the current hand before showing a summary. Continuing keeps this table and run intact.
      </p>
    </div>
  );
}

export function AnalystRail({
  idPrefix = 'rail',
  tab,
  onTabChange,
  feedback,
  recentHands,
  settings,
  pendingSettings,
  onSettingsChange,
  fullDepths,
  manifest,
  sessionDecisions,
  historyWarning,
  opponentModel,
}: AnalystRailProps) {
  return (
    <aside className="overflow-hidden rounded-lg border border-border bg-surface shadow-sm" aria-label="Table analyst">
      <div className="grid grid-cols-4 border-b border-border" role="tablist" aria-label="Analyst panels">
        {TABS.map(({ value, label, icon: Icon }) => (
          <button
            key={value}
            type="button"
            role="tab"
            aria-selected={tab === value}
            aria-controls={`${idPrefix}-${value}`}
            onClick={() => onTabChange(value)}
            className={`min-h-14 px-1 text-[10px] font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent ${tab === value ? 'bg-surface-2 text-fg' : 'text-muted hover:text-fg'}`}
          >
            <Icon className="mx-auto mb-1 h-4 w-4" aria-hidden="true" />
            {label}
          </button>
        ))}
      </div>

      <div id={`${idPrefix}-${tab}`} role="tabpanel" className="max-h-[min(680px,70vh)] overflow-y-auto">
        {historyWarning && (
          <p className="border-b border-amber-500/30 bg-amber-500/10 px-4 py-2 text-xs">{historyWarning}</p>
        )}
        {tab === 'feedback' && <FeedbackPanel feedback={feedback} />}
        {tab === 'history' && <HistoryPanel recentHands={recentHands} />}
        {tab === 'settings' && (
          <>
            <SettingsPanel
              settings={settings}
              pendingSettings={pendingSettings}
              onSettingsChange={onSettingsChange}
              fullDepths={fullDepths}
            />
            {opponentModel && manifest?.runtime?.kind === 'neural-deep-cfr-v1' && (
              <div className="border-t border-border p-4 text-xs leading-5">
                <p className="font-semibold">Local opponent evidence</p>
                <p className="mt-1 text-muted">
                  {opponentModel.observations} observations ·{' '}
                  {opponentModel.stableEvidence} stable · confidence{' '}
                  {pct(opponentModel.confidence)}
                </p>
                <p className="mt-1 text-muted">
                  Current response weight {pct(opponentModel.responseWeight)} (cap{' '}
                  {pct(opponentModel.maximumResponseWeight)}). No hand data leaves this browser.
                </p>
              </div>
            )}
          </>
        )}
        {tab === 'stats' && <StatsPanel decisions={sessionDecisions} />}
      </div>

      {manifest && (
        <details className="border-t border-border p-4 text-xs">
          <summary className="flex min-h-11 cursor-pointer items-center font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent">
            Model & assumptions
          </summary>
          <div className="mt-3 space-y-2 leading-5 text-muted">
            <p>{manifest.label} · {manifest.version}</p>
            {manifest.runtime?.kind === 'neural-deep-cfr-v1' && (
              <p>
                Frozen Deep CFR baseline plus a confidence-capped exploit response. Weights are immutable static artifacts; opponent evidence stays in local IndexedDB.
              </p>
            )}
            {manifest.runtime?.kind === 'rust-continual-resolver-v1' && (
              <p>
                The server replays this hand through the pinned Rust policy and resolves each postflop decision from exact public ranges. Missing weights or a failed resolve pause the table; no fallback strategy is scored.
              </p>
            )}
            {manifest.validation.exploitabilityGateDeferred && (
              <p className="font-semibold text-amber-700 dark:text-amber-300">
                Experimental: the exploitability release gate is deferred. This model is not labeled Approximate GTO.
              </p>
            )}
            <p>
              {manifest.abstraction.blindsBb.join('/')}bb blinds · {manifest.abstraction.rake} rake · {manifest.abstraction.recall} recall
            </p>
            <p>{manifest.abstraction.actionSizing}</p>
            <p>{manifest.abstraction.cardAbstraction}</p>
            {manifest.validation.exploitabilityEstimateBb !== undefined && (
              <p>
                Estimated exploitability {manifest.validation.exploitabilityEstimateBb.toFixed(3)}bb/hand
                {manifest.validation.exploitabilityUpper99Bb !== undefined
                  ? ` · 99% upper ${manifest.validation.exploitabilityUpper99Bb.toFixed(3)}bb/hand`
                  : ''}
              </p>
            )}
            {manifest.validation.crossSeedFrequencyMae !== undefined && (
              <p>
                Cross-seed MAE {(manifest.validation.crossSeedFrequencyMae * 100).toFixed(1)}% · primary agreement {((manifest.validation.primaryActionAgreement ?? 0) * 100).toFixed(1)}%
              </p>
            )}
            {manifest.validation.policyCoverage !== undefined && (
              <p>
                Lookup coverage {(manifest.validation.policyCoverage * 100).toFixed(3)}% · precise action-EV coverage {((manifest.validation.actionEvStandardErrorCoverage ?? 0) * 100).toFixed(1)}%
              </p>
            )}
            {manifest.validation.projectedStorageBytes !== undefined && (
              <p>
                Projected hosted policy {(manifest.validation.projectedStorageBytes / 1024 ** 3).toFixed(2)}GiB
              </p>
            )}
            {manifest.validation.notes.map((note) => <p key={note}>{note}</p>)}
          </div>
        </details>
      )}
    </aside>
  );
}
