import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import catalog from '@/data/solved-spots/catalog.json';
import { evaluate } from '@/lib/evaluator';
import { seededRandom } from '@/lib/practice-engine';
import { advanceSpot, playSpotOpponent, sessionNode, spotActions, spotCombo, SPOT_SEATS,
  startSpot, validateSolvedSpot, type SolvedSpot, type SpotCatalogEntry, type SpotSession } from '@/lib/solved-spots';

const entry = catalog.find((e) => e.board.length === 4) as SpotCatalogEntry;
const bytes = () => readFileSync(path.join(process.cwd(),'public',entry.url));
const spot = () => validateSolvedSpot(JSON.parse(bytes().toString()));

/** Force a legal public river in tests only; the engine burns before drawing. */
function withRiver(session: SpotSession, card: number): SpotSession {
  const burn = session.state.deck.find((c) => c !== card)!;
  return { ...session, state: { ...session.state,
    deck: [...session.state.deck.filter((c) => c !== card && c !== burn),card,burn] } };
}

describe('complete turn-to-river saved game', () => {
  it('rejects missing chance cards, mismatched public boards and false precision metadata', () => {
    const broken = JSON.parse(bytes().toString()) as SolvedSpot;
    broken.nodes.flatMap((n) => n.actions).find((a) => a.nextByRiver)!.nextByRiver!.pop();
    expect(() => validateSolvedSpot(broken)).toThrow('invalid');
    const wrongBoard = JSON.parse(bytes().toString()) as SolvedSpot;
    wrongBoard.nodes.find((n) => n.board?.length === 5)!.board![0] = 0;
    expect(() => validateSolvedSpot(wrongBoard)).toThrow('invalid');
    const wrongCount = JSON.parse(bytes().toString()) as SolvedSpot;
    wrongCount.verification.riverOutcomes = 1;
    expect(() => validateSolvedSpot(wrongCount)).toThrow('invalid');
  });

  it('plays every public decision and all legal chance edges with exact chip conservation', () => {
    const solved = spot(), visited = new Set<string>(), riverCards = new Set<number>();
    function walk(session: SpotSession) {
      const s = session.state;
      expect(s.stacksBb['big-blind']+s.stacksBb['button-small-blind']+s.potBb
        +s.streetBetsBb['big-blind']+s.streetBetsBb['button-small-blind']).toBeCloseTo(100,8);
      expect(new Set([...s.board,...Object.values(s.holeCards).flat()]).size).toBe(s.board.length+4);
      if (s.terminal) { expect(session.nodeId).toBeNull(); return; }
      const node = sessionNode(solved,session); visited.add(node.id);
      for (const action of node.actions) {
        if (action.nextByRiver) {
          const holes = Object.values(s.holeCards).flat();
          for (const [card] of action.nextByRiver) if (!holes.includes(card)) {
            riverCards.add(card);
            const next = advanceSpot(solved,withRiver(session,card),action.id);
            expect(next.state.board).toEqual([...solved.input.board,card]);
            walk(next);
          }
        } else walk(advanceSpot(solved,session,action.id));
      }
    }
    for (let seed=0; seed<20 && visited.size < solved.nodes.length; seed++) {
      walk(startSpot(solved,entry.sha256,'big-blind',seededRandom(seed)));
    }
    expect(visited.size).toBe(solved.nodes.length);
    expect(riverCards.size).toBe(48);
  });

  it('does not let the hidden runout affect turn actions or EVs', () => {
    const solved = spot(), session = startSpot(solved,entry.sha256,'big-blind',seededRandom(3));
    expect(spotActions(solved,session)).toEqual(spotActions(solved,{ ...session,state: {
      ...session.state,deck: [...session.state.deck].reverse(),
    } }));
    expect(session.state.street).toBe('turn');
    expect(session.state.board.length).toBe(4);
    expect(() => advanceSpot(solved,session,'off-tree')).toThrow('not part');
  });

  it('runs consecutive opponent actions when BB calls the turn and acts first on the river', () => {
    const solved = spot();
    for (let seed=0; seed<100; seed++) {
      const first = startSpot(solved,entry.sha256,'button-small-blind',seededRandom(seed));
      const checked = advanceSpot(solved,first,'check');
      const bet = sessionNode(solved,checked).actions.find((a) => a.kind === 'bet')!;
      const facing = advanceSpot(solved,checked,bet.id);
      const actions = spotActions(solved,facing), index = actions.findIndex((a) => a.kind === 'call');
      if (index < 0 || actions[index].probability <= 0) continue;
      const roll = actions.slice(0,index).reduce((s,a) => s+a.probability,0)+actions[index].probability/2;
      let calls=0;
      const next = playSpotOpponent(solved,facing,() => calls++ === 0 ? roll : 0);
      expect(next.state.actionHistory.slice(-2).map((a) => a.actor)).toEqual(['big-blind','big-blind']);
      expect(next.state.actionHistory.slice(-2).map((a) => a.street)).toEqual(['turn','river']);
      expect(next.state.toAct).toBe('button-small-blind');
      expect(calls).toBe(2);
      return;
    }
    throw new Error('No compatible frozen turn call found');
  });

  it('independently checks every EV and both whole-subgame best responses, with no river peek', () => {
    const solved = spot();
    const nodes = new Map(solved.nodes.map((n) => [n.id,n]));
    const rows = new Map(solved.nodes.flatMap((n) => n.rows.map((r) => [`${n.id}|${r.combo}`,r] as const)));
    const combos = [...new Set(solved.ranges.flat().map(([key]) => key))];
    const cards = new Map(combos.map((key) => [key,spotCombo(key)]));
    const rivers = Array.from({length:52},(_,c) => c).filter((c) => !solved.input.board.includes(c));
    const strengths = new Map(rivers.map((river) => [river,new Map(combos.filter((key) => !cards.get(key)!.includes(river))
      .map((key) => [key,evaluate([...solved.input.board,river,...cards.get(key)!])]))]));
    const equityCache = new Map<string,number>();
    function equity(a: number,b: number,river?: number): number {
      if (river !== undefined) return Math.sign(strengths.get(river)!.get(a)!-strengths.get(river)!.get(b)!);
      const id = `${a}|${b}`, cached = equityCache.get(id);
      if (cached !== undefined) return cached;
      const legal = rivers.filter((c) => !cards.get(a)!.includes(c) && !cards.get(b)!.includes(c));
      expect(legal.length).toBe(44);
      const value = legal.reduce((sum,c) => sum+equity(a,b,c),0)/44;
      equityCache.set(id,value); return value;
    }
    let maxError = 0;
    function walk(id: string,own: number,combo: number,opponents: Array<[number,number]>,br: boolean): number {
      const node = nodes.get(id)!, river = node.board![4];
      const values = node.actions.map((action,i) => {
        const nextOpponents: Array<[number,number]> = node.actor === own ? opponents
          : opponents.map(([key,w]) => [key,w*rows.get(`${id}|${key}`)!.probabilities[i]]);
        if (action.nextNodeId) return walk(action.nextNodeId,own,combo,nextOpponents,br);
        if (action.nextByRiver) return action.nextByRiver.reduce((sum,[card,nextId]) => {
          if (cards.get(combo)!.includes(card)) return sum;
          const masked: Array<[number,number]> = nextOpponents.filter(([key]) => !cards.get(key)!.includes(card)).map(([key,w]) => [key,w/44]);
          return sum+walk(nextId,own,combo,masked,br);
        },0);
        const invested = [...node.investedBb];
        if (action.kind === 'call') invested[node.actor] += Math.min(solved.input.depthBb-invested[node.actor],
          Math.max(...node.streetBetsBb)-node.streetBetsBb[node.actor]);
        return nextOpponents.reduce((sum,[other,w]) => sum+w*(action.kind === 'fold'
          ? node.actor === own ? -invested[own] : invested[1-own]
          : equity(combo,other,river)*invested[own]),0);
      });
      if (node.actor !== own) return values.reduce((s,v) => s+v,0);
      if (br) return Math.max(...values); // Max after all hidden hands AND unrevealed river chance.
      const row = rows.get(`${id}|${combo}`)!, mass = opponents.reduce((s,[,w]) => s+w,0);
      if (mass>0) values.forEach((v,i) => { maxError=Math.max(maxError,Math.abs(v/mass-row.evBb![i])); });
      return values.reduce((s,v,i) => s+v*row.probabilities[i],0);
    }
    const profile=[0,0], response=[0,0]; let joint=0;
    for (let p=0; p<2; p++) for (const [combo,w] of solved.ranges[p]) {
      const opponents = solved.ranges[1-p].filter(([key]) => !cards.get(combo)!.some((c) => cards.get(key)!.includes(c)));
      if (p===0) joint += w*opponents.reduce((s,[,weight]) => s+weight,0);
      profile[p] += w*walk(solved.rootNodeId,p,combo,opponents,false);
      response[p] += w*walk(solved.rootNodeId,p,combo,opponents,true);
    }
    for (let p=0; p<2; p++) {
      expect(profile[p]/joint).toBeCloseTo(solved.verification.profileValueBb[p],9);
      expect(response[p]/joint).toBeCloseTo(solved.verification.bestResponseValueBb[p],9);
    }
    expect(maxError).toBeLessThan(.0001);
    expect((response[0]+response[1]-profile[0]-profile[1])/joint).toBeLessThan(.05);
  });
});
