'use client';

import { useEffect, useMemo, useState, useCallback } from 'react';
import { Card, cardToStr } from '@/lib/cards';
import { weightsToRange, serializeRange } from '@/lib/cards';
import { CardSlots } from '@/components/board/CardPicker';
import { RangeEditor } from '@/components/range/RangeEditor';
import { EquityPanel } from '@/components/equity/EquityPanel';
import { AiPanel } from '@/components/ai/AiPanel';
import { useEquity, PlayerSpec } from '@/lib/equity/client';
import type { EquityResult } from '@/lib/equity/compute';
import type { SpotContext } from '@/lib/ai/prompt';

type Mode = 'hand' | 'range';
const HERO_COLOR = 'rgb(var(--call))';
const VILL_COLOR = 'rgb(var(--raise))';

interface Side {
  mode: Mode;
  hand: Card[];
  weights: Record<string, number>;
}

const emptySide = (): Side => ({ mode: 'hand', hand: [], weights: {} });

export default function CalculatorPage() {
  const [hero, setHero] = useState<Side>(emptySide);
  const [villain, setVillain] = useState<Side>(() => ({
    mode: 'range',
    hand: [],
    weights: {},
  }));
  const [board, setBoard] = useState<Card[]>([]);
  const [result, setResult] = useState<EquityResult | null>(null);
  const { run, running } = useEquity();

  const used = useMemo(() => {
    const s = new Set<Card>(board);
    hero.hand.forEach((c) => s.add(c));
    villain.hand.forEach((c) => s.add(c));
    return s;
  }, [board, hero.hand, villain.hand]);

  const toSpec = useCallback((side: Side): PlayerSpec | null => {
    if (side.mode === 'hand') {
      if (side.hand.length !== 2) return null;
      return { hand: [side.hand[0], side.hand[1]] };
    }
    const range = weightsToRange(side.weights);
    if (range.size === 0) return null;
    return { range };
  }, []);

  const calculate = useCallback(async () => {
    const a = toSpec(hero);
    const b = toSpec(villain);
    if (!a || !b) {
      setResult(null);
      return;
    }
    const r = await run([a, b], board, 200000);
    setResult(r);
  }, [hero, villain, board, run, toSpec]);

  // Auto-recalculate (debounced) whenever inputs change.
  useEffect(() => {
    const t = setTimeout(calculate, 250);
    return () => clearTimeout(t);
  }, [calculate]);

  const buildSpot = useCallback((): SpotContext | null => {
    const describe = (s: Side) =>
      s.mode === 'hand'
        ? s.hand.map(cardToStr).join('')
        : serializeRange(weightsToRange(s.weights)) || '(empty)';
    return {
      kind: 'equity',
      description: 'Heads-up equity spot from the calculator.',
      board: board.map(cardToStr).join('') || 'preflop',
      heroRange: describe(hero),
      villainRange: describe(villain),
      equity: result
        ? { hero: result.equities[0], villain: result.equities[1] }
        : undefined,
    };
  }, [hero, villain, board, result]);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-xl font-semibold">Equity Calculator</h1>
        <p className="text-sm text-muted">
          Hand vs hand, hand vs range, or range vs range — with exact
          enumeration when the runout is small and Monte Carlo otherwise.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_320px_1fr]">
        <SidePanel
          title="Hero"
          color={HERO_COLOR}
          side={hero}
          onChange={setHero}
          used={used}
        />

        <div className="flex flex-col gap-4">
          <div className="rounded-lg border border-border bg-surface p-4">
            <div className="mb-2 text-sm font-semibold">Board</div>
            <CardSlots
              count={5}
              cards={board}
              used={used}
              onChange={setBoard}
            />
            <div className="mt-2 flex gap-2 text-xs text-muted">
              <button className="hover:text-fg" onClick={() => setBoard([])}>
                Clear board
              </button>
            </div>
          </div>

          <EquityPanel
            result={result}
            running={running}
            labels={['Hero', 'Villain']}
            colors={[HERO_COLOR, VILL_COLOR]}
          />

          <button
            onClick={calculate}
            className="rounded-md bg-accent px-4 py-2 text-sm font-semibold text-accent-fg hover:opacity-90"
          >
            {running ? 'Calculating…' : 'Recalculate'}
          </button>

          <AiPanel getSpot={buildSpot} />
        </div>

        <SidePanel
          title="Villain"
          color={VILL_COLOR}
          side={villain}
          onChange={setVillain}
          used={used}
        />
      </div>
    </div>
  );
}

function SidePanel({
  title,
  color,
  side,
  onChange,
  used,
}: {
  title: string;
  color: string;
  side: Side;
  onChange: (s: Side) => void;
  used: Set<Card>;
}) {
  // Cards used by other slots (exclude this side's own hand so it can re-pick).
  const usedForThis = useMemo(() => {
    const s = new Set(used);
    side.hand.forEach((c) => s.delete(c));
    return s;
  }, [used, side.hand]);

  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2 font-semibold">
          <span className="h-3 w-3 rounded-full" style={{ background: color }} />
          {title}
        </div>
        <div className="inline-flex rounded-md border border-border p-0.5 text-xs">
          {(['hand', 'range'] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => onChange({ ...side, mode: m })}
              className={
                'rounded px-2 py-0.5 capitalize transition-colors ' +
                (side.mode === m
                  ? 'bg-surface-2 text-fg'
                  : 'text-muted hover:text-fg')
              }
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      {side.mode === 'hand' ? (
        <CardSlots
          count={2}
          cards={side.hand}
          used={usedForThis}
          onChange={(hand) => onChange({ ...side, hand })}
          label="Hole cards"
        />
      ) : (
        <RangeEditor
          weights={side.weights}
          onChange={(weights) => onChange({ ...side, weights })}
          accent={color}
        />
      )}
    </div>
  );
}
