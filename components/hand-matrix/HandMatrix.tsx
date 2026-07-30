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
  /** optional accessible/hover detail that is not rendered inside the cell */
  cellDescription?: (label: string) => string | undefined;
  onCellClick?: (label: string) => void;
  /** display mode: currently inspected hand class */
  selectedLabel?: string;
  className?: string;
}

const ACCENT = 'rgb(var(--call))';

export function HandMatrix({
  mode,
  weights = {},
  onWeightsChange,
  strategy = {},
  annotation,
  cellDescription,
  onCellClick,
  selectedLabel,
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
    if (mode !== 'select') return;
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

  const handleKeyDown = (label: string, e: React.KeyboardEvent) => {
    if (mode !== 'select') return;
    if (e.key !== 'Enter' && e.key !== ' ') return;
    e.preventDefault();
    const current = weights[label] ?? 0;
    onWeightsChange?.({ ...weights, [label]: current > 0 ? 0 : 1 });
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
            const isSelected = mode === 'display' && selectedLabel === label;
            const cellClass =
              'relative flex items-center justify-center overflow-hidden rounded-[3px] border text-[10px] font-medium leading-none transition-[border-color,box-shadow,background-color] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ' +
              (isPair ? 'border-border/80 ' : 'border-border/40 ') +
              (isSelected
                ? 'z-10 ring-2 ring-accent ring-offset-1 ring-offset-surface '
                : '') +
              'bg-surface-2';
            const annotationText = annotation?.(label);
            const descriptionText = cellDescription?.(label);
            const title = [label, annotationText, descriptionText]
              .filter(Boolean)
              .join(' - ');
            const content = (
              <>
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
                      ? 'bg-black/70 px-0.5 py-px text-white'
                      : w > 0.4
                        ? 'text-white/95'
                        : 'text-fg/80')
                  }
                >
                  {label}
                </span>
                {annotationText && (
                  <span className="pointer-events-none absolute bottom-[1px] right-[2px] z-10 text-[7px] text-white/80 drop-shadow-[0_1px_1px_rgba(0,0,0,0.6)]">
                    {annotationText}
                  </span>
                )}
              </>
            );
            if (mode === 'display' && !onCellClick) {
              return (
                <div
                  key={label}
                  aria-label={title}
                  className={cellClass}
                  title={title}
                  style={{ minWidth: 0 }}
                >
                  {content}
                </div>
              );
            }
            return (
              <button
                key={label}
                type="button"
                onMouseDown={(event) => handleDown(label, event)}
                onMouseEnter={() => handleEnter(label)}
                onKeyDown={(event) => handleKeyDown(label, event)}
                onClick={
                  mode === 'display' ? () => onCellClick?.(label) : undefined
                }
                {...(mode === 'select'
                  ? { 'aria-pressed': w > 0 }
                  : { 'aria-pressed': isSelected })}
                aria-label={
                  mode === 'select'
                    ? `${label}${w > 0 ? ', selected' : ''}`
                    : title
                }
                className={cellClass}
                title={title}
                style={{ minWidth: 0 }}
              >
                {content}
              </button>
            );
          })
        )}
      </div>
    </div>
  );
}
