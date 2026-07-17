'use client';

import { useCallback, useId, useMemo, useState } from 'react';
import { Card, cardToStr, weightsToRange, serializeRange } from '@/lib/cards';
import { CardSlots } from '@/components/board/CardPicker';
import { RangeEditor } from '@/components/range/RangeEditor';
import { AiPanel } from '@/components/ai/AiPanel';
import { SolverResults } from '@/components/solver/SolverResults';
import {
  useSolver,
  rangeToTriples,
  SolverResult,
  SolverInput,
} from '@/lib/solver/client';
import type { SpotContext } from '@/lib/ai/prompt';

const OOP_COLOR = 'rgb(var(--check))';
const IP_COLOR = 'rgb(var(--allin))';

export default function SolverPage() {
  const [board, setBoard] = useState<Card[]>([]);
  const [oop, setOop] = useState<Record<string, number>>({});
  const [ip, setIp] = useState<Record<string, number>>({});
  const [pot, setPot] = useState(6);
  const [stack, setStack] = useState(100);
  const [betSizes, setBetSizes] = useState('33, 75');
  const [raiseSizes, setRaiseSizes] = useState('100');
  const [iterations, setIterations] = useState(300);
  const [result, setResult] = useState<SolverResult | null>(null);
  const { solve, running, available } = useSolver();

  const used = useMemo(() => new Set<Card>(board), [board]);

  const parseSizes = (s: string) =>
    s
      .split(/[,\s]+/)
      .map((x) => parseFloat(x))
      .filter((x) => isFinite(x) && x > 0)
      .map((x) => x / 100);

  const runSolve = useCallback(async () => {
    const oopRange = weightsToRange(oop);
    const ipRange = weightsToRange(ip);
    if (oopRange.size === 0 || ipRange.size === 0 || board.length < 3) return;

    const input: SolverInput = {
      board,
      oop: rangeToTriples(oopRange),
      ip: rangeToTriples(ipRange),
      pot,
      stack,
      bet_sizes: parseSizes(betSizes),
      raise_sizes: parseSizes(raiseSizes),
      iterations,
      max_combos: 200,
    };
    const r = await solve(input);
    setResult(r);
  }, [oop, ip, board, pot, stack, betSizes, raiseSizes, iterations, solve]);

  const buildSpot = useCallback((): SpotContext | null => {
    if (board.length < 3) return null;
    return {
      kind: 'postflop',
      description: 'Postflop spot solved with in-browser CFR (all-in-equity model).',
      board: board.map(cardToStr).join(''),
      heroRange: serializeRange(weightsToRange(oop)) || '(empty)',
      villainRange: serializeRange(weightsToRange(ip)) || '(empty)',
      potBB: pot,
      stackBB: stack,
      extra: {
        'Bet sizes (% pot)': betSizes,
        ...(result
          ? {
              Exploitability: `${result.exploitability_pct}% of pot`,
              'OOP EV': `${result.oop_ev} bb`,
              'IP EV': `${result.ip_ev} bb`,
            }
          : {}),
      },
    };
  }, [board, oop, ip, pot, stack, betSizes, result]);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-xl font-semibold">Postflop Solver</h1>
        <p className="max-w-3xl text-sm text-muted">
          Real Discounted-CFR (CFR+) solving, compiled from Rust to WebAssembly
          and run entirely in your browser. Single-street, all-in-equity model —
          produces GTO bet/check/call/fold frequencies and EVs with observable
          exploitability convergence.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_360px_1fr]">
        <div className="rounded-lg border border-border bg-surface p-4">
          <div className="mb-3 font-semibold" style={{ color: OOP_COLOR }}>
            OOP range
          </div>
          <RangeEditor weights={oop} onChange={setOop} accent={OOP_COLOR} />
        </div>

        <div className="flex flex-col gap-4">
          <div className="rounded-lg border border-border bg-surface p-4">
            <div className="mb-2 text-sm font-semibold">Board</div>
            <CardSlots count={5} cards={board} used={used} onChange={setBoard} />

            <div className="mt-4 grid grid-cols-2 gap-3">
              <NumberField label="Pot (bb)" value={pot} onChange={setPot} />
              <NumberField label="Eff. stack (bb)" value={stack} onChange={setStack} />
            </div>
            <div className="mt-3 grid grid-cols-2 gap-3">
              <TextField label="Bet sizes (% pot)" value={betSizes} onChange={setBetSizes} />
              <TextField label="Raise sizes (% pot)" value={raiseSizes} onChange={setRaiseSizes} />
            </div>
            <div className="mt-3">
              <label
                htmlFor="solver-iterations"
                className="mb-1 block text-xs text-muted"
              >
                Iterations: {iterations}
              </label>
              <input
                id="solver-iterations"
                type="range"
                min={50}
                max={1000}
                step={50}
                value={iterations}
                onChange={(e) => setIterations(parseInt(e.target.value))}
                className="w-full accent-[color:rgb(var(--accent))]"
              />
            </div>

            <button
              onClick={runSolve}
              disabled={board.length < 3 || running || !available}
              className="mt-4 w-full rounded-md bg-accent px-4 py-2 text-sm font-semibold text-accent-fg hover:opacity-90 disabled:opacity-40"
            >
              {running ? 'Solving…' : 'Solve'}
            </button>
            {board.length < 3 && (
              <p className="mt-2 text-center text-xs text-muted">
                Set at least a flop (3 cards).
              </p>
            )}
            {!available && (
              <p className="mt-2 text-center text-xs text-muted">
                Loading solver engine…
              </p>
            )}
          </div>

          <AiPanel getSpot={buildSpot} />
        </div>

        <div className="rounded-lg border border-border bg-surface p-4">
          <div className="mb-3 font-semibold" style={{ color: IP_COLOR }}>
            IP range
          </div>
          <RangeEditor weights={ip} onChange={setIp} accent={IP_COLOR} />
        </div>
      </div>

      {result && <SolverResults result={result} />}
    </div>
  );
}

function NumberField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
}) {
  const id = useId();
  return (
    <div>
      <label htmlFor={id} className="mb-1 block text-xs text-muted">
        {label}
      </label>
      <input
        id={id}
        type="number"
        inputMode="decimal"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        className="w-full rounded-md border border-border bg-surface-2 p-2 text-sm outline-none focus:border-accent focus-visible:ring-2 focus-visible:ring-accent"
      />
    </div>
  );
}

function TextField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (s: string) => void;
}) {
  const id = useId();
  return (
    <div>
      <label htmlFor={id} className="mb-1 block text-xs text-muted">
        {label}
      </label>
      <input
        id={id}
        inputMode="decimal"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-md border border-border bg-surface-2 p-2 font-mono text-xs outline-none focus:border-accent focus-visible:ring-2 focus-visible:ring-accent"
      />
    </div>
  );
}
