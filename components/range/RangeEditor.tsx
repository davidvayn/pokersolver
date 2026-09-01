'use client';

import { useMemo } from 'react';
import { HandMatrix } from '@/components/hand-matrix/HandMatrix';
import {
  weightsToRange,
  rangeComboCount,
  handClassLabel,
} from '@/lib/cards';

interface RangeEditorProps {
  weights: Record<string, number>;
  onChange: (weights: Record<string, number>) => void;
  title?: string;
  accent?: string;
  compact?: boolean;
  showActions?: boolean;
}

export function RangeEditor({
  weights,
  onChange,
  title,
  accent,
  compact = false,
  showActions = true,
}: RangeEditorProps) {
  const comboCount = useMemo(
    () => rangeComboCount(weightsToRange(weights)),
    [weights]
  );
  const pct = ((comboCount / 1326) * 100).toFixed(1);

  function selectAll() {
    const all: Record<string, number> = {};
    for (let r = 0; r < 13; r++)
      for (let c = 0; c < 13; c++) all[handClassLabel(r, c)] = 1;
    onChange(all);
  }

  return (
    <div className={`flex min-h-0 flex-col ${compact ? 'gap-2' : 'gap-3'}`}>
      {title && (
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 font-medium">
            {accent && (
              <span
                className="h-3 w-3 rounded-full"
                style={{ background: accent }}
              />
            )}
            {title}
          </div>
          <div className="text-xs text-muted">
            {comboCount.toFixed(0)} combos · {pct}%
          </div>
        </div>
      )}

      <HandMatrix mode="select" weights={weights} onWeightsChange={onChange} />

      {showActions && (
        <div className="flex justify-end gap-1">
          <button
            type="button"
            onClick={selectAll}
            className="min-h-8 rounded-md border border-border px-2 text-xs text-muted transition-colors hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            All
          </button>
          <button
            type="button"
            onClick={() => onChange({})}
            className="min-h-8 rounded-md border border-border px-2 text-xs text-muted transition-colors hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            Clear
          </button>
        </div>
      )}
    </div>
  );
}
