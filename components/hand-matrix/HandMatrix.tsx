'use client';

import { useCallback, useRef, useState } from 'react';
import { handClassLabel } from '@/lib/cards';

export interface StrategySegment {
  /** CSS color, e.g. 'rgb(var(--raise))' */
  color: string;
  /** fraction 0..1 of the cell height this action occupies */
  fraction: number;
  label?: string;
}

export interface HandMatrixProps {
  mode: 'select' | 'display';
  /** select mode: weight 0..1 per hand-class label */
  weights?: Record<string, number>;
  onWeightsChange?: (next: Record<string, number>) => void;
  /** display mode: stacked strategy segments per hand-class label */
  strategy?: Record<string, StrategySegment[]>;
  /** optional per-cell numeric annotation shown small (e.g. EV or frequency) */
  annotation?: (label: string) => string | undefined;
  onCellClick?: (label: string) => void;
  className?: string;
}

const ACCENT = 'rgb(var(--call))';

export function HandMatrix({
  mode,
  weights = {},
  onWeightsChange,
  strategy = {},
  annotation,
  onCellClick,
  className,
}: HandMatrixProps) {
  const [painting, setPainting] = useState<null | number>(null); // target weight while dragging
  const draftRef = useRef<Record<string, number>>({});

  const applyPaint = useCallback(
    (label: string, value: number) => {
      draftRef.current[label] = value;
      onWeightsChange?.({ ...weights, ...draftRef.current });
    },
    [onWeightsChange, weights]
  );

  const handleDown = (label: string, e: React.MouseEvent) => {
    if (mode !== 'select') {
      onCellClick?.(label);
      return;
    }
    e.preventDefault();
    const current = weights[label] ?? 0;
    const target = current > 0 ? 0 : 1; // toggle
    draftRef.current = {};
    setPainting(target);
    applyPaint(label, target);
  };

  const handleEnter = (label: string) => {
    if (mode === 'select' && painting !== null) applyPaint(label, painting);
  };

  const endPaint = () => {
    if (painting !== null) {
      draftRef.current = {};
      setPainting(null);
    }
  };

  return (
    <div
      className={'select-none ' + (className ?? '')}
      onMouseLeave={endPaint}
      onMouseUp={endPaint}
    >
      <div
        className="grid aspect-square w-full gap-[2px]"
        style={{ gridTemplateColumns: 'repeat(13, minmax(0, 1fr))' }}
      >
        {Array.from({ length: 13 }).map((_, row) =>
          Array.from({ length: 13 }).map((__, col) => {
            const label = handClassLabel(row, col);
            const isPair = row === col;
            const w = weights[label] ?? 0;
            const segs = strategy[label];
            return (
              <button
                key={label}
                onMouseDown={(e) => handleDown(label, e)}
                onMouseEnter={() => handleEnter(label)}
                className={
                  'relative flex items-center justify-center overflow-hidden rounded-[3px] border text-[10px] font-medium leading-none transition-colors ' +
                  (isPair
                    ? 'border-border/80 '
                    : 'border-border/40 ') +
                  'bg-surface-2'
                }
                title={label + (annotation?.(label) ? ` · ${annotation(label)}` : '')}
                style={{ minWidth: 0 }}
              >
                {/* Fill layer */}
                {mode === 'select' ? (
                  <span
                    className="pointer-events-none absolute inset-0"
                    style={{
                      background: ACCENT,
                      opacity: w === 0 ? 0 : 0.25 + 0.6 * w,
                    }}
                  />
                ) : segs && segs.length ? (
                  <span className="pointer-events-none absolute inset-0 flex flex-col-reverse">
                    {segs.map((s, i) => (
                      <span
                        key={i}
                        style={{
                          height: `${s.fraction * 100}%`,
                          background: s.color,
                        }}
                      />
                    ))}
                  </span>
                ) : null}

                <span
                  className={
                    'pointer-events-none relative z-10 ' +
                    (mode === 'display' && segs?.length
                      ? 'text-white/95 drop-shadow-[0_1px_1px_rgba(0,0,0,0.5)]'
                      : w > 0.4
                        ? 'text-white/95'
                        : 'text-fg/80')
                  }
                >
                  {label}
                </span>
                {annotation?.(label) && (
                  <span className="pointer-events-none absolute bottom-[1px] right-[2px] z-10 text-[7px] text-white/80 drop-shadow-[0_1px_1px_rgba(0,0,0,0.6)]">
                    {annotation(label)}
                  </span>
                )}
              </button>
            );
          })
        )}
      </div>
    </div>
  );
}
