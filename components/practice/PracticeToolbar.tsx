'use client';

import { ChevronDown } from 'lucide-react';
import { useId, useState } from 'react';
import { Select } from '@/components/ui/Select';
import { pushFoldDepths } from '@/lib/push-fold-policy';
import type { PracticeSettings } from '@/lib/practice-types';

export function PracticeToolbar({
  settings,
  pendingSettings,
  fullDepths,
  onChange,
}: {
  settings: PracticeSettings;
  pendingSettings: PracticeSettings | null;
  fullDepths: number[];
  onChange: (settings: PracticeSettings) => void;
}) {
  const controlsId = useId();
  const [expanded, setExpanded] = useState(false);
  const shown = pendingSettings ?? settings;
  const patch = (next: Partial<PracticeSettings>) =>
    onChange({ ...shown, ...next });
  const depths = shown.mode === 'push-fold' ? pushFoldDepths() : fullDepths;
  const controls = (
    <div id={controlsId} className="practice-toolbar" data-open={expanded}>
      <Select
        label="Mode"
        value={shown.mode}
        options={[
          { value: 'full-hand', label: 'Full hand' },
          { value: 'preflop', label: 'Preflop' },
          { value: 'postflop', label: 'Postflop' },
          { value: 'push-fold', label: 'Push/fold' },
        ]}
        onChange={(mode) => patch({ mode: mode as PracticeSettings['mode'] })}
      />
      <Select
        label="Stack"
        value={String(
          shown.mode === 'push-fold' ? shown.pushFoldDepthBb : shown.depthBb
        )}
        options={
          depths.length
            ? depths.map((depth) => ({
                value: String(depth),
                label: `${depth} bb`,
              }))
            : [{ value: '20', label: 'No validated model' }]
        }
        disabled={!depths.length}
        onChange={(value) =>
          shown.mode === 'push-fold'
            ? patch({
                pushFoldDepthBb: Number(
                  value
                ) as PracticeSettings['pushFoldDepthBb'],
              })
            : patch({ depthBb: Number(value) as PracticeSettings['depthBb'] })
        }
      />
      <Select
        label="Seat"
        value={shown.heroSeat}
        options={[
          { value: 'alternate', label: 'Alternate' },
          { value: 'button-small-blind', label: 'BTN / SB' },
          { value: 'big-blind', label: 'Big blind' },
        ]}
        onChange={(heroSeat) =>
          patch({ heroSeat: heroSeat as PracticeSettings['heroSeat'] })
        }
      />
      <Select
        label="Goal"
        value={String(shown.decisionGoal)}
        options={[
          { value: 'continuous', label: 'Continuous' },
          ...[25, 50, 100].map((goal) => ({
            value: String(goal),
            label: `${goal} decisions`,
          })),
        ]}
        onChange={(value) =>
          patch({
            decisionGoal:
              value === 'continuous'
                ? 'continuous'
                : (Number(value) as 25 | 50 | 100),
          })
        }
      />
    </div>
  );
  const modeLabel = {
    'full-hand': 'Full hand',
    preflop: 'Preflop',
    postflop: 'Postflop',
    'push-fold': 'Push/fold',
  }[shown.mode];
  return (
    <div className="practice-config">
      <button
        type="button"
        className="practice-config-toggle"
        aria-label="Configure practice"
        aria-expanded={expanded}
        aria-controls={controlsId}
        onClick={() => setExpanded(!expanded)}
      >
        <span>
          {modeLabel} ·{' '}
          {shown.mode === 'push-fold' ? shown.pushFoldDepthBb : shown.depthBb}{' '}
          bb
        </span>
        <span className="inline-flex items-center gap-2">
          Configure{' '}
          <ChevronDown
            className={`h-4 w-4 ${expanded ? 'rotate-180' : ''}`}
            aria-hidden="true"
          />
        </span>
      </button>
      {controls}
    </div>
  );
}
