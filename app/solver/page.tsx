'use client';

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
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
import { SolverWorkspace } from '@/components/solver/SolverWorkspace';
import {
  useSolver,
  rangeToTriples,
} from '@/lib/solver/client';
import type { SolverResult, SolverInput } from '@/lib/solver/client';
import type { SpotContext } from '@/lib/ai/prompt';
import { useUi } from '@/lib/ui-store';

// A ready-to-solve single-raised-pot example (BTN opens, BB calls) so the page
// works the moment it loads and there's always a known-good spot to fall back
// to. IP = Button opening range, OOP = Big Blind defending range.
const EXAMPLE = {
  oop: '99-22,AJs-A8s,KTs+,QTs+,JTs,T9s,98s,87s,76s,AQo,AJo,KQo',
  ip: 'TT-22,ATs+,A5s,KTs+,QTs+,J9s+,T9s,98s,AQo+,KQo',
  board: 'Qh7s2c',
};
const exampleWeights = (s: string) => rangeToWeights(parseRange(s));

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
  const [result, setResult] = useState<SolverResult | null>(null);
  const solveVersion = useRef(0);
  const { solve, running, available, error: solverError } = useSolver();
  const showSolverStats = useUi((state) => state.showSolverStats);

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
      description:
        'Postflop spot solved with an in-browser CFR+ one-street all-in-equity abstraction.',
      board: board.map(cardToStr).join(''),
      heroRange: serializeRange(weightsToRange(oop)) || '(empty)',
      villainRange: serializeRange(weightsToRange(ip)) || '(empty)',
      potBB: pot,
      stackBB: stack,
      extra: {
        'Bet sizes (% pot)': betSizes,
        'Raise sizes (% pot)': raiseSizes,
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
  }, [board, oop, ip, pot, stack, betSizes, raiseSizes, result]);

  return (
    <SolverWorkspace
      board={board}
      used={used}
      onBoardChange={setBoard}
      oop={oop}
      ip={ip}
      onOopChange={setOop}
      onIpChange={setIp}
      pot={pot}
      stack={stack}
      betSizes={betSizes}
      raiseSizes={raiseSizes}
      onPotChange={setPot}
      onStackChange={setStack}
      onBetSizesChange={setBetSizes}
      onRaiseSizesChange={setRaiseSizes}
      result={result}
      running={running}
      available={available}
      solverError={solverError}
      missing={missing}
      showSolverStats={showSolverStats}
      getAnalysisSpot={buildSpot}
      onClear={clearAll}
    />
  );
}
