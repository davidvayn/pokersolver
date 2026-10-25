import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import catalog from '@/data/solved-spots/catalog.json';
import { comboKey } from '@/lib/cards';
import { evaluate } from '@/lib/evaluator';
import { seededRandom } from '@/lib/practice-engine';
import { advanceSpot, loadSolvedSpot, playSpotOpponent, sessionNode, spotActions, spotCombo,
  spotEvLoss, SPOT_SEATS, startSpot, validateSolvedSpot, type SolvedSpot, type SpotSession, type SpotCatalogEntry } from '@/lib/solved-spots';

describe.each(catalog.filter((e) => e.board.length === 5))('saved river: $title', (rawEntry) => {
  const entry = rawEntry as SpotCatalogEntry;
  const bytes = readFileSync(path.join(process.cwd(), 'public', entry.url));
  const spot = validateSolvedSpot(JSON.parse(bytes.toString()));
  it('pins bytes, checks the checksum and fails closed on corrupt/missing files', async () => {
    const fetcher = vi.spyOn(globalThis, 'fetch');
    try {
      fetcher.mockResolvedValue(new Response(bytes));
      expect((await loadSolvedSpot(entry)).input.id).toBe(entry.id);
      fetcher.mockResolvedValue(new Response(Buffer.from('{}')));
      await expect(loadSolvedSpot(entry)).rejects.toThrow('checksum');
      fetcher.mockResolvedValue(new Response('', { status: 404 }));
      await expect(loadSolvedSpot(entry)).rejects.toThrow('retry');
    } finally { fetcher.mockRestore(); }
  });

  it('rejects missing rows, invalid sums and unqualified artifacts', () => {
    const bad = JSON.parse(bytes.toString()) as SolvedSpot;
    bad.nodes[0].rows.pop();
    expect(() => validateSolvedSpot(bad)).toThrow('invalid');
    const badSum = JSON.parse(bytes.toString()) as SolvedSpot;
    badSum.nodes[0].rows[0].probabilities[0] = 2;
    expect(() => validateSolvedSpot(badSum)).toThrow('invalid');
    expect(() => validateSolvedSpot({ ...spot, verification: { ...spot.verification, accepted: false } })).toThrow();
    expect(() => validateSolvedSpot({ ...spot, verification: { ...spot.verification, totalBestResponseGainBb: NaN } })).toThrow();
  });

  it('preserves the canonical combo keys and samples only compatible range deals', () => {
    for (let key=0; key<1326; key++) expect(comboKey(...spotCombo(key))).toBe(key);
    for (let seed=0; seed<100; seed++) {
      const session = startSpot(spot, entry.sha256, SPOT_SEATS[seed % 2], seededRandom(seed));
      const state = session.state;
      expect(new Set([...state.board, ...Object.values(state.holeCards).flat()]).size).toBe(9);
      for (let p=0; p<2; p++) expect(spot.ranges[p].some(([key]) => key === comboKey(...state.holeCards[SPOT_SEATS[p]]))).toBe(true);
      expect(state.potBb).toBe(spot.input.potBb);
      expect(state.stacksBb['big-blind']).toBe(spot.input.depthBb-spot.input.potBb/2);
    }
  });

  it('plays EVERY saved branch with the shared hand engine and conserves chips', () => {
    const visited = new Set<string>();
    const kinds = new Set<string>();
    let terminals = 0;
    function walk(session: SpotSession) {
      const state = session.state;
      expect(state.stacksBb['big-blind'] + state.stacksBb['button-small-blind'] + state.potBb
        + state.streetBetsBb['big-blind'] + state.streetBetsBb['button-small-blind']).toBeCloseTo(spot.input.depthBb*2,8);
      if (state.terminal) { terminals++; expect(session.nodeId).toBeNull(); return; }
      const node = sessionNode(spot,session);
      expect(visited.has(node.id)).toBe(false);
      visited.add(node.id);
      for (const action of node.actions) { kinds.add(action.kind); walk(advanceSpot(spot,session,action.id)); }
    }
    walk(startSpot(spot,entry.sha256,'big-blind',seededRandom(7)));
    expect(visited.size).toBe(spot.nodes.length);
    expect(terminals).toBeGreaterThan(10);
    expect([...kinds]).toEqual(expect.arrayContaining(['fold','check','call','bet','raise','all-in']));
  });

  it('samples opponent frequencies, disallows off-tree actions and does not guess missing rows', () => {
    const session = startSpot(spot,entry.sha256,'button-small-blind',seededRandom(3));
    const next = playSpotOpponent(spot,session,seededRandom(6));
    expect(next.state.toAct).toBe('button-small-blind');
    expect(next.state.actionHistory.length).toBe(1);
    expect(() => advanceSpot(spot,session,'raise-to-999')).toThrow('not part');
    expect(() => spotActions(spot,{ ...session, state: { ...session.state, board: [0,1,2,3,4] } })).toThrow('outside');
    const withoutRows = { ...spot, nodes: spot.nodes.map((node) => ({ ...node, rows: [] })) };
    expect(() => spotActions(withoutRows,session)).toThrow('missing');
  });

  it('grades actual EV loss, not the popularity of a mixed action', () => {
    const actions = spotActions(spot,startSpot(spot,entry.sha256,'big-blind',seededRandom(3)));
    const tied = actions.slice(0,2).map((a,i) => ({ ...a, evBb: 1.5, probability: i ? .01 : .99 }));
    expect(spotEvLoss(tied,tied[1].id)).toBe(0);
    expect(spotEvLoss([{ ...tied[0], evBb: 2 }, { ...tied[1], evBb: 1.5 }],tied[1].id)).toBe(.5);
    expect(spotEvLoss([{ ...tied[0], evBb: null }, tied[1]],tied[1].id)).toBeNull();
  });

  it('independently reproduces all EVs and information-set best responses in TypeScript', () => {
    // Deliberately separate from Rust card marginals, evaluator and betting state.
    // At own decisions choose one action AFTER integrating all opponent hands.
    const nodes = new Map(spot.nodes.map((node) => [node.id,node]));
    const rows = new Map(spot.nodes.flatMap((n) => n.rows.map((r) => [`${n.id}|${r.combo}`,r] as const)));
    const strengths = new Map([...new Set(spot.ranges.flat().map(([key]) => key))]
      .map((key) => [key,evaluate([...spot.input.board,...spotCombo(key)])]));
    let maxError = 0;
    function walk(session: SpotSession, own: number, combo: number, opponents: Array<[number,number]>, br: boolean): number {
      const { state } = session;
      if (state.terminal) return opponents.reduce((sum,[key,w]) => {
        const win = state.result?.reason === 'fold'
          ? state.result.winner === SPOT_SEATS[own] ? 1 : -1
          : Math.sign(strengths.get(combo)! - strengths.get(key)!);
        const value = win > 0 ? state.totalCommittedBb[SPOT_SEATS[1-own]]
          : win < 0 ? -state.totalCommittedBb[SPOT_SEATS[own]] : 0;
        return sum + w * value;
      },0);
      const node = nodes.get(session.nodeId!)!;
      const values = node.actions.map((action,i) => walk(advanceSpot(spot,session,action.id),own,combo,
        node.actor === own ? opponents : opponents.map(([key,w]) => [key,w * rows.get(`${node.id}|${key}`)!.probabilities[i]]),br));
      if (node.actor !== own) return values.reduce((s,v) => s+v,0);
      if (br) return Math.max(...values);
      const row = rows.get(`${node.id}|${combo}`)!;
      const mass = opponents.reduce((s,[,w]) => s+w,0);
      if (mass > 0) values.forEach((value,i) => { maxError = Math.max(maxError,Math.abs(value/mass - row.evBb![i])); });
      return values.reduce((s,v,i) => s+v*row.probabilities[i],0);
    }
    let joint = 0;
    const profile = [0,0], br = [0,0];
    const session = startSpot(spot,entry.sha256,'big-blind',seededRandom(7));
    for (let p=0; p<2; p++) for (const [key,w] of spot.ranges[p]) {
      const cards = spotCombo(key);
      const opponents = spot.ranges[1-p].filter(([other]) => !spotCombo(other).some((c) => cards.includes(c)));
      if (p === 0) joint += w * opponents.reduce((s,[,weight]) => s+weight,0);
      profile[p] += w * walk(session,p,key,opponents,false);
      br[p] += w * walk(session,p,key,opponents,true);
    }
    for (let p=0; p<2; p++) {
      expect(profile[p]/joint).toBeCloseTo(spot.verification.profileValueBb[p],10);
      expect(br[p]/joint).toBeCloseTo(spot.verification.bestResponseValueBb[p],10);
    }
    expect(maxError).toBeLessThan(.0001);
    expect((br[0]+br[1]-profile[0]-profile[1])/joint).toBeLessThan(.05);
  });
});
