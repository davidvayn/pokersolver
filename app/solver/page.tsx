'use client';

import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import {
  Card,
  cardToStr,
  weightsToRange,
  serializeRange,
  parseRange,
  parseBoard,
  rangeToWeights,
} from '@/lib/cards';
import { CardSlots } from '@/components/board/CardPicker';
import { RangeEditor } from '@/components/range/RangeEditor';
import { AiPanel } from '@/components/ai/AiPanel';
import {
  SolverNerdStats,
  StrategyView,
} from '@/components/solver/SolverResults';
import {
  useSolver,
  rangeToTriples,
} from '@/lib/solver/client';
import type { SolverResult, SolverInput } from '@/lib/solver/client';
import type { SpotContext } from '@/lib/ai/prompt';
import { useUi } from '@/lib/ui-store';

const OOP_COLOR = 'rgb(var(--check))';
const IP_COLOR = 'rgb(var(--allin))';

// A ready-to-solve single-raised-pot example (BTN opens, BB calls) so the page
// works the moment it loads and there's always a known-good spot to fall back
// to. IP = Button opening range, OOP = Big Blind defending range.
const EXAMPLE = {
  oop: '99-22,AJs-A8s,KTs+,QTs+,JTs,T9s,98s,87s,76s,AQo,AJo,KQo',
  ip: 'TT-22,ATs+,A5s,KTs+,QTs+,J9s+,T9s,98s,AQo+,KQo',
  board: 'Qh7s2c',
};
const exampleWeights = (s: string) => rangeToWeights(parseRange(s));

// Premade pot/stack/bet/raise configurations, plus a fully custom option.
interface Sizing {
  label: string;
  pot?: number;
  stack?: number;
  bet?: string;
  raise?: string;
}
const SIZING_PRESETS: Record<string, Sizing> = {
  srp: { label: 'Single-raised pot · 33/75', pot: 6, stack: 100, bet: '33, 75', raise: '100' },
  threebet: { label: '3-bet pot · 33/66', pot: 18, stack: 100, bet: '33, 66', raise: '100' },
  small: { label: 'Small bets · 25/50', pot: 6, stack: 100, bet: '25, 50', raise: '100' },
  big: { label: 'Big bets · 75/125', pot: 6, stack: 100, bet: '75, 125', raise: '75' },
  custom: { label: 'Custom…' },
};

const SOLVE_ITERATIONS = 1000;
const SOLVER_ALGORITHM = 'cfr_plus' as const;
const SOLVER_LABEL = 'Approximate GTO · CFR+';

export default function SolverPage() {
  const [board, setBoard] = useState<Card[]>(() => parseBoard(EXAMPLE.board));
  const [oop, setOop] = useState<Record<string, number>>(() =>
    exampleWeights(EXAMPLE.oop)
  );
  const [ip, setIp] = useState<Record<string, number>>(() =>
    exampleWeights(EXAMPLE.ip)
  );
  const [pot, setPot] = useState(6);
  const [stack, setStack] = useState(100);
  const [betSizes, setBetSizes] = useState('33, 75');
  const [raiseSizes, setRaiseSizes] = useState('100');
  const [sizing, setSizing] = useState('srp');
  const [result, setResult] = useState<SolverResult | null>(null);
  const [rangeTab, setRangeTab] = useState<'oop' | 'ip'>('oop');
  const [stratTab, setStratTab] = useState<'oop' | 'ip'>('oop');
  const solveVersion = useRef(0);
  const { solve, running, available, error: solverError } = useSolver();
  const showSolverStats = useUi((state) => state.showSolverStats);

  function applySizing(key: string) {
    setSizing(key);
    const p = SIZING_PRESETS[key];
    if (key !== 'custom' && p.pot !== undefined) {
      setPot(p.pot);
      setStack(p.stack!);
      setBetSizes(p.bet!);
      setRaiseSizes(p.raise!);
    }
  }

  const used = useMemo(() => new Set<Card>(board), [board]);

  // What (if anything) is stopping a solve, for clear inline feedback.
  const missing = useMemo(() => {
    if (Object.values(oop).every((w) => !w)) return 'Add an OOP range';
    if (Object.values(ip).every((w) => !w)) return 'Add an IP range';
    if (board.length < 3) return 'Set at least a flop (3 cards)';
    return null;
  }, [oop, ip, board]);
  const ready = missing === null;

  function clearAll() {
    solveVersion.current++;
    setOop({});
    setIp({});
    setBoard([]);
    setResult(null);
  }

  const parseSizes = (s: string) =>
    s
      .split(/[,\s]+/)
      .map((x) => parseFloat(x))
      .filter((x) => isFinite(x) && x > 0)
      .map((x) => x / 100);

  const runSolve = useCallback(async (version: number) => {
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
      iterations: SOLVE_ITERATIONS,
      algorithm: SOLVER_ALGORITHM,
      max_combos: 200,
    };
    const r = await solve(input);
    if (version === solveVersion.current) setResult(r);
  }, [oop, ip, board, pot, stack, betSizes, raiseSizes, solve]);

  // Auto-solve (debounced) whenever the spot is valid and inputs change — so
  // the page shows a result on load and updates as you edit, no hunting for a
  // button.
  useEffect(() => {
    if (!ready || !available) return;
    const version = ++solveVersion.current;
    const t = setTimeout(() => runSolve(version), 500);
    return () => clearTimeout(t);
  }, [ready, available, runSolve]);

  const buildSpot = useCallback((): SpotContext | null => {
    if (board.length < 3) return null;
    return {
      kind: 'postflop',
      description: 'Postflop spot solved with in-browser CFR+ (one-street all-in-equity abstraction).',
      board: board.map(cardToStr).join(''),
      heroRange: serializeRange(weightsToRange(oop)) || '(empty)',
      villainRange: serializeRange(weightsToRange(ip)) || '(empty)',
      potBB: pot,
      stackBB: stack,
      extra: {
        'Bet sizes (% pot)': betSizes,
        'Solver model': SOLVER_LABEL,
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
      <div className="flex items-center gap-3">
        <h1 className="text-lg font-semibold">Postflop Solver</h1>
        {running && (
          <span role="status" className="text-xs text-muted">
            Solving…
          </span>
        )}
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,340px)_minmax(0,1fr)]">
        {/* Setup (left) — range editor */}
        <div className="flex flex-col gap-6">
          <div className="rounded-lg border border-border bg-surface p-4">
            <div className="mb-3 flex items-center gap-1 rounded-md bg-surface-2 p-1">
              <TabButton
                active={rangeTab === 'oop'}
                color={OOP_COLOR}
                onClick={() => setRangeTab('oop')}
              >
                OOP range
              </TabButton>
              <TabButton
                active={rangeTab === 'ip'}
                color={IP_COLOR}
                onClick={() => setRangeTab('ip')}
              >
                IP range
              </TabButton>
            </div>
            <p className="mb-3 text-xs text-muted">
              {rangeTab === 'oop'
                ? 'Out of position — acts first. Drag the grid or use a preset.'
                : 'In position — acts last. Drag the grid or use a preset.'}
            </p>
            {rangeTab === 'oop' ? (
              <RangeEditor weights={oop} onChange={setOop} accent={OOP_COLOR} />
            ) : (
              <RangeEditor weights={ip} onChange={setIp} accent={IP_COLOR} />
            )}
          </div>

          {/* Board + bet sizing, under the ranges */}
          <div className="rounded-lg border border-border bg-surface p-4">
            <div className="mb-1.5 text-xs text-muted">Board</div>
            <CardSlots
              count={5}
              cards={board}
              used={used}
              onChange={setBoard}
              size="lg"
            />

            <label
              htmlFor="sizing"
              className="mb-1.5 mt-4 block text-xs text-muted"
            >
              Bet sizing
            </label>
            <select
              id="sizing"
              value={sizing}
              onChange={(e) => applySizing(e.target.value)}
              className="w-full rounded-md border border-border bg-surface-2 p-2 text-sm text-fg outline-none focus:border-accent focus-visible:ring-2 focus-visible:ring-accent"
            >
              {Object.entries(SIZING_PRESETS).map(([k, v]) => (
                <option key={k} value={k}>
                  {v.label}
                </option>
              ))}
            </select>

            {sizing === 'custom' && (
              <div className="mt-3 grid grid-cols-2 gap-3">
                <NumberField label="Pot (bb)" value={pot} onChange={setPot} />
                <NumberField
                  label="Stack (bb)"
                  value={stack}
                  onChange={setStack}
                />
                <TextField
                  label="Bet %"
                  value={betSizes}
                  onChange={setBetSizes}
                />
                <TextField
                  label="Raise %"
                  value={raiseSizes}
                  onChange={setRaiseSizes}
                />
              </div>
            )}
          </div>
        </div>

        {/* Solution (right) — the enlarged strategy chart */}
        <div>
          {result && !result.error ? (
            <div className="flex flex-col gap-4">
              <div className="rounded-lg border border-border bg-surface p-4">
                <div className="mb-3 flex items-center gap-2">
                  <div className="flex flex-1 gap-1 rounded-md bg-surface-2 p-1">
                    <TabButton
                      active={stratTab === 'oop'}
                      color={OOP_COLOR}
                      onClick={() => setStratTab('oop')}
                    >
                      OOP · first to act
                    </TabButton>
                    <TabButton
                      active={stratTab === 'ip'}
                      color={IP_COLOR}
                      onClick={() => setStratTab('ip')}
                    >
                      IP · vs check
                    </TabButton>
                  </div>
                  <button
                    onClick={clearAll}
                    className="rounded-md border border-border px-3 py-1.5 text-sm font-medium text-muted hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                  >
                    Clear
                  </button>
                </div>
                <StrategyView
                  node={stratTab === 'oop' ? result.oop : result.ip}
                />
                {result.truncated && (
                  <p className="mt-3 text-xs leading-relaxed text-muted">
                    Ranges were capped to keep the solve fast; results use the
                    highest-weight combos.
                  </p>
                )}
              </div>
              {showSolverStats && <SolverNerdStats result={result} />}
            </div>
          ) : result?.error || solverError ? (
            <div
              role="alert"
              className="rounded-lg border border-raise/40 bg-raise/10 p-4 text-sm text-raise"
            >
              <p>Solver error: {result?.error ?? solverError}</p>
              {solverError && (
                <button
                  type="button"
                  onClick={() => window.location.reload()}
                  className="mt-3 min-h-11 rounded-md border border-raise/40 px-3 py-2 font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-raise"
                >
                  Reload solver
                </button>
              )}
            </div>
          ) : (
            <div className="grid h-64 place-items-center rounded-lg border border-dashed border-border bg-surface p-8 text-center text-sm text-muted">
              {running
                ? 'Solving…'
                : !available
                  ? 'Initializing solver…'
                : missing
                  ? `${missing} to see the solution`
                  : 'Set up a spot to see the solution'}
            </div>
          )}
        </div>
      </div>

      <AiPanel getSpot={buildSpot} />
    </div>
  );
}

function TabButton({
  active,
  color,
  onClick,
  children,
}: {
  active: boolean;
  color: string;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      aria-pressed={active}
      className={
        'flex-1 rounded px-3 py-1.5 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ' +
        (active ? 'bg-surface text-fg shadow-sm' : 'text-muted hover:text-fg')
      }
    >
      <span className="inline-flex items-center gap-1.5">
        <span className="h-2 w-2 rounded-full" style={{ background: color }} />
        {children}
      </span>
    </button>
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
