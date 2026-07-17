'use client';

import { useMemo, useState } from 'react';
import { HandMatrix } from '@/components/hand-matrix/HandMatrix';
import {
  parseRange,
  serializeRange,
  rangeToWeights,
  weightsToRange,
  rangeComboCount,
  handClassLabel,
} from '@/lib/cards';

interface RangeEditorProps {
  weights: Record<string, number>;
  onChange: (weights: Record<string, number>) => void;
  title?: string;
  accent?: string;
}

const PRESETS: { label: string; range: string }[] = [
  { label: 'BTN RFI', range: '22+,A2s+,K7s+,Q9s+,J9s+,T8s+,97s+,86s+,75s+,64s+,54s,A7o+,K9o+,Q9o+,J9o+,T9o' },
  { label: 'CO RFI', range: '22+,A2s+,K9s+,QTs+,J9s+,T9s,98s,A9o+,KTo+,QJo' },
  { label: 'UTG RFI', range: '77+,AJs+,KQs,AKo' },
  { label: 'Broadways', range: 'AKs,AQs,AJs,ATs,KQs,KJs,QJs,JTs,AKo,AQo,AJo,KQo' },
];

export function RangeEditor({ weights, onChange, title, accent }: RangeEditorProps) {
  const [text, setText] = useState('');
  const [editingText, setEditingText] = useState(false);

  const comboCount = useMemo(
    () => rangeComboCount(weightsToRange(weights)),
    [weights]
  );
  const pct = ((comboCount / 1326) * 100).toFixed(1);

  const displayText = editingText ? text : serializeRange(weightsToRange(weights));

  function applyText(value: string) {
    try {
      const w = rangeToWeights(parseRange(value));
      onChange(w);
    } catch {
      /* ignore parse errors while typing */
    }
  }

  function selectAll() {
    const all: Record<string, number> = {};
    for (let r = 0; r < 13; r++)
      for (let c = 0; c < 13; c++) all[handClassLabel(r, c)] = 1;
    onChange(all);
  }

  return (
    <div className="flex flex-col gap-3">
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

      <div className="flex flex-wrap gap-1">
        {PRESETS.map((p) => (
          <button
            key={p.label}
            onClick={() => onChange(rangeToWeights(parseRange(p.range)))}
            className="rounded-md border border-border px-2 py-1 text-xs text-muted hover:text-fg"
          >
            {p.label}
          </button>
        ))}
        <button
          onClick={selectAll}
          className="rounded-md border border-border px-2 py-1 text-xs text-muted hover:text-fg"
        >
          All
        </button>
        <button
          onClick={() => onChange({})}
          className="rounded-md border border-border px-2 py-1 text-xs text-muted hover:text-fg"
        >
          Clear
        </button>
      </div>

      <textarea
        value={displayText}
        onFocus={() => {
          setEditingText(true);
          setText(serializeRange(weightsToRange(weights)));
        }}
        onBlur={() => setEditingText(false)}
        onChange={(e) => {
          setText(e.target.value);
          applyText(e.target.value);
        }}
        rows={2}
        spellCheck={false}
        placeholder="e.g. 22+, AJs+, KQo, A5s-A2s"
        className="w-full resize-none rounded-md border border-border bg-surface-2 p-2 font-mono text-xs text-fg outline-none focus:border-accent"
      />
    </div>
  );
}
