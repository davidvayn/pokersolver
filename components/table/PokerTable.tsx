'use client';

import { useCallback, useEffect, useState } from 'react';
import {
  Position,
  TableFormat,
  positionLabelForSeats,
} from '@/lib/positions';

interface PokerTableProps {
  format: TableFormat;
  hero: Position;
  villain?: Position;
  onHero: (p: Position) => void;
  onVillain?: (p: Position) => void;
  /** show the villain assignment control + seat */
  withVillain?: boolean;
}

export function PokerTable({
  format,
  hero,
  villain,
  onHero,
  onVillain,
  withVillain = true,
}: PokerTableProps) {
  const [assigning, setAssigning] = useState<'hero' | 'villain'>('hero');
  const positions = format.positions;
  const positionLabel = (position: Position) =>
    positionLabelForSeats(position, format.seats);

  const assign = useCallback(
    (p: Position) => {
      if (assigning === 'villain' && withVillain) {
        if (p === hero) return; // can't be both
        onVillain?.(p);
      } else {
        if (withVillain && p === villain) onVillain?.(hero); // swap
        onHero(p);
      }
    },
    [assigning, withVillain, hero, villain, onHero, onVillain]
  );

  // Keyboard: 1..9 select a seat for the active role — routed through assign()
  // so the same hero/villain conflict guard and swap logic apply as on click.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement)
        return;
      const n = parseInt(e.key, 10);
      if (!isNaN(n) && n >= 1 && n <= positions.length) {
        assign(positions[n - 1]);
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [assign, positions]);

  return (
    <div className="flex flex-col gap-3">
      {withVillain && (
        <div className="flex items-center gap-2 text-xs">
          <span className="text-muted">Assign:</span>
          <Segmented
            options={[
              { key: 'hero', label: 'Hero', color: 'rgb(var(--call))' },
              { key: 'villain', label: 'Villain', color: 'rgb(var(--raise))' },
            ]}
            value={assigning}
            onChange={(v) => setAssigning(v as 'hero' | 'villain')}
          />
          <span className="ml-auto text-muted">Press 1–{positions.length} to switch</span>
        </div>
      )}

      {/* Oval table with seats around it */}
      <div className="relative mx-auto aspect-[16/9] w-full max-w-md">
        <div
          className="absolute inset-[14%] rounded-[50%] border-4 shadow-inner"
          style={{
            background: 'rgb(var(--felt))',
            borderColor: 'rgb(0 0 0 / 0.25)',
          }}
        >
          <div className="grid h-full w-full place-items-center text-center text-accent-fg/70">
            <div className="text-[11px] uppercase tracking-widest">
              {format.seats}-max
            </div>
          </div>
        </div>
        {positions.map((p, i) => {
          const angle = (i / positions.length) * 2 * Math.PI - Math.PI / 2;
          const x = 50 + 46 * Math.cos(angle);
          const y = 50 + 46 * Math.sin(angle);
          const isHero = p === hero;
          const isVillain = withVillain && p === villain;
          return (
            <button
              key={p}
              type="button"
              onClick={() => assign(p)}
              aria-pressed={isHero || isVillain}
              aria-label={`${positionLabel(p)}${isHero ? ', hero' : isVillain ? ', villain' : ''}`}
              className={
                'absolute min-h-11 min-w-11 -translate-x-1/2 -translate-y-1/2 rounded-full border px-2 py-1 text-[11px] font-semibold shadow-card transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ' +
                (isHero
                  ? 'text-call-fg'
                  : isVillain
                    ? 'text-raise-fg'
                    : 'bg-surface text-fg')
              }
              style={{
                left: `${x}%`,
                top: `${y}%`,
                background: isHero
                  ? 'rgb(var(--call))'
                  : isVillain
                    ? 'rgb(var(--raise))'
                    : undefined,
                borderColor: isHero || isVillain ? 'transparent' : 'rgb(var(--border))',
              }}
              title={positionLabel(p)}
            >
              <span className="mr-1 text-[8px] opacity-60">{i + 1}</span>
              {positionLabel(p)}
            </button>
          );
        })}
      </div>

      {/* Quick-switch pills */}
      <div className="flex flex-wrap justify-center gap-1">
        {positions.map((p) => {
          const isHero = p === hero;
          const isVillain = withVillain && p === villain;
          return (
            <button
              key={p}
              type="button"
              onClick={() => assign(p)}
              aria-pressed={isHero || isVillain}
              aria-label={`${positionLabel(p)}${isHero ? ', hero' : isVillain ? ', villain' : ''}`}
              className={
                'min-h-11 rounded-md border px-3 py-2 text-xs transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ' +
                (isHero
                  ? 'border-transparent text-call-fg'
                  : isVillain
                    ? 'border-transparent text-raise-fg'
                    : 'border-border text-muted hover:text-fg')
              }
              style={{
                background: isHero
                  ? 'rgb(var(--call))'
                  : isVillain
                    ? 'rgb(var(--raise))'
                    : undefined,
              }}
            >
              {positionLabel(p)}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function Segmented({
  options,
  value,
  onChange,
}: {
  options: { key: string; label: string; color: string }[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="inline-flex rounded-md border border-border p-0.5">
      {options.map((o) => (
        <button
          key={o.key}
          type="button"
          onClick={() => onChange(o.key)}
          aria-pressed={value === o.key}
          className={
            'min-h-11 rounded px-3 py-2 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ' +
            (value === o.key ? '' : 'text-muted hover:text-fg')
          }
          style={{
            background: value === o.key ? o.color : undefined,
            color:
              value === o.key
                ? o.key === 'hero'
                  ? 'rgb(var(--call-fg))'
                  : 'rgb(var(--raise-fg))'
                : undefined,
          }}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
