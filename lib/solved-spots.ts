import { comboKey, comboLabelToCombos, handClassLabel, type Combo } from '@/lib/cards';
import { applyAction } from '@/lib/practice-engine';
import type { HandState, LegalAction, PolicyAction, Seat } from '@/lib/practice-types';

export interface SpotVerification {
  method: string;
  compatibleDeals: number;
  profileValueBb: [number, number];
  bestResponseValueBb: [number, number];
  unilateralGainBb: [number, number];
  totalBestResponseGainBb: number;
  maximumActionEvErrorBb: number;
  maximumProbabilitySumError: number;
  zeroSumResidualBb: number;
  accepted: boolean;
  riverOutcomes?: number;
}
export interface SpotCatalogEntry {
  id: string; title: string; board: number[]; depthBb: number;
  url: string; sha256: string; bytes: number; verification: SpotVerification;
}
export interface SolvedSpot {
  schema: 'solved-river-spot-v1' | 'solved-turn-river-spot-v1';
  inputSha256: string;
  input: {
    id: string; title: string; description: string; rangeAssumptions: string;
    board: number[]; depthBb: number; potBb: number;
    rangeClasses: [string[], string[]]; betPotFractions: number[];
    rangeClassWeights?: [number[], number[]];
    raisePotFractions: number[]; raiseCap: number; iterations: number;
  };
  rootNodeId: string;
  ranges: [Array<[number, number]>, Array<[number, number]>];
  nodes: SolvedSpotNode[];
  verification: SpotVerification;
}
export interface SolvedSpotNode {
  id: string; actor: 0 | 1; investedBb: [number, number]; streetBetsBb: [number, number];
  actions: Array<LegalAction & { nextNodeId: string | null; nextByRiver?: Array<[number,string]> }>;
  rows: Array<{ combo: number; probabilities: number[]; evBb: number[] | null }>;
  board?: number[];
}
export interface SpotSession { state: HandState; nodeId: string | null }
export const SPOT_SEATS: readonly Seat[] = ['button-small-blind', 'big-blind'];

export function spotCombo(key: number): Combo {
  if (!Number.isInteger(key) || key < 0 || key >= 1326) throw new Error('Invalid combo key');
  const high = Math.floor((1 + Math.sqrt(1 + 8 * key)) / 2);
  return [high, key - high * (high - 1) / 2];
}
const compatible = (first: Combo, second: Combo) => !first.some((card) => second.includes(card));
const near = (a: number, b: number) => Number.isFinite(a) && Math.abs(a - b) < 1e-6;
const handClasses = new Set(Array.from({length:169},(_,i) => handClassLabel(Math.floor(i/13),i%13)));

/** Inclusion multipliers are not action frequencies or normalized range shares. */
export function spotRangeSummary(input: SolvedSpot['input'], player: 0 | 1): string {
  const weights = input.rangeClassWeights?.[player];
  return input.rangeClasses[player].map((label,i) => weights ? `${label} ×${weights[i]}` : label).join(', ');
}

/** Declared per-combo inclusion weights, normalized after public card removal. */
function declaredRanges(input: SolvedSpot['input']): Map<number,number>[] {
  const weights = input.rangeClassWeights;
  if (!Array.isArray(input.rangeClasses) || input.rangeClasses.length !== 2
    || (weights !== undefined && (!Array.isArray(weights) || weights.length !== 2))) throw new Error();
  return input.rangeClasses.map((labels,p) => {
    if (!Array.isArray(labels) || !labels.length || new Set(labels).size !== labels.length
      || labels.some((label) => !handClasses.has(label))
      || (weights !== undefined && (!Array.isArray(weights[p]) || weights[p].length !== labels.length
        || weights[p].some((w) => !Number.isFinite(w) || w <= 0 || w > 1)))) throw new Error();
    const range = new Map<number,number>();
    labels.forEach((label,i) => {
      for (const combo of comboLabelToCombos(label)) if (!combo.some((c) => input.board.includes(c))) {
        range.set(comboKey(...combo),weights?.[p][i] ?? 1);
      }
    });
    const sum = [...range.values()].reduce((s,w) => s+w,0);
    if (!Number.isFinite(sum) || sum <= 0) throw new Error();
    const normalized = new Map([...range].map(([key,w]) => [key,w/sum]));
    if ([...normalized.values()].some((w) => w <= 0)) throw new Error();
    return normalized;
  });
}

/** Structural checks in addition to the content hash and offline exact checker. */
export function validateSolvedSpot(value: unknown): SolvedSpot {
  const spot = value as SolvedSpot;
  try {
    const v = spot.verification;
    const turnRoot = spot.schema === 'solved-turn-river-spot-v1';
    if ((!turnRoot && spot.schema !== 'solved-river-spot-v1') || !/^[a-f0-9]{64}$/.test(spot.inputSha256)
      || !v.accepted || v.method !== (turnRoot ? 'direct-compatible-pair-and-river-enumeration-of-exported-f32-policy-v1' : 'direct-compatible-pair-enumeration-of-exported-f32-policy-v1')
      || !Number.isFinite(v.totalBestResponseGainBb) || v.totalBestResponseGainBb < 0 || v.totalBestResponseGainBb > .05
      || !Number.isFinite(v.maximumActionEvErrorBb) || v.maximumActionEvErrorBb < 0 || v.maximumActionEvErrorBb > .0001
      || !Number.isFinite(v.maximumProbabilitySumError) || v.maximumProbabilitySumError > 1e-6
      || !Number.isFinite(v.zeroSumResidualBb) || v.zeroSumResidualBb > 1e-5) throw new Error();
    const input = spot.input;
    if (!Number.isFinite(input.depthBb) || input.depthBb < 2 || input.depthBb > 1000
      || !Number.isFinite(input.potBb) || input.potBb < 2 || input.potBb >= input.depthBb * 2
      || input.board.length !== (turnRoot ? 4 : 5) || new Set(input.board).size !== input.board.length
      || input.board.some((c) => !Number.isInteger(c) || c < 0 || c > 51)
      || spot.ranges.length !== 2 || spot.nodes.length < 1 || spot.nodes.length > 4096
      || !Number.isInteger(input.raiseCap) || input.raiseCap < 0 || input.raiseCap > 1
      || [input.betPotFractions,input.raisePotFractions].some((sizes) => !sizes.length || sizes.length > 2
        || sizes.some((f,i) => !Number.isFinite(f) || f <= 0 || f > 4 || (i>0 && f <= sizes[i-1])))) throw new Error();
    const declared = declaredRanges(input);
    for (const [p,range] of spot.ranges.entries()) {
      if (!range.length || new Set(range.map(([key]) => key)).size !== range.length
        || range.length !== declared[p].size
        || range.some(([key, weight]) => !Number.isFinite(weight) || weight <= 0 || weight > 1
          || spotCombo(key).some((card) => input.board.includes(card))
          || !declared[p].has(key) || Math.abs(weight-declared[p].get(key)!) > 1e-12)
        || !near(range.reduce((sum, [,w]) => sum + w, 0), 1)) throw new Error();
    }
    const ids = new Set(spot.nodes.map((node) => node.id));
    const nodes = new Map(spot.nodes.map((node) => [node.id,node]));
    if (ids.size !== spot.nodes.length || !ids.has(spot.rootNodeId)) throw new Error();
    if (turnRoot && (!Number.isSafeInteger(v.riverOutcomes) || v.riverOutcomes !== v.compatibleDeals * 44
      || JSON.stringify(nodes.get(spot.rootNodeId)?.board) !== JSON.stringify(input.board))) throw new Error();
    for (const node of spot.nodes) {
      const board = node.board ?? input.board;
      if (turnRoot ? (!node.board || ![4,5].includes(board.length)
          || board.some((c,i) => !Number.isInteger(c) || c < 0 || c > 51 || (i<4 && c!==input.board[i]))
          || new Set(board).size !== board.length) : node.board !== undefined) throw new Error();
      if (![0,1].includes(node.actor) || !node.actions.length || node.actions.length > 8
        || new Set(node.actions.map((a) => a.id)).size !== node.actions.length
        || node.investedBb.length !== 2 || node.streetBetsBb.length !== 2
        || node.investedBb.some((a) => !Number.isFinite(a) || a < 0 || a > input.depthBb)
        || node.streetBetsBb.some((a) => !Number.isFinite(a) || a < 0 || a > input.depthBb)) throw new Error();
      for (const action of node.actions) {
        if (!['fold','check','call','bet','raise','all-in'].includes(action.kind)
          || typeof action.label !== 'string' || typeof action.id !== 'string'
          || (action.nextNodeId !== null && !ids.has(action.nextNodeId))
          || (['bet','raise','all-in'].includes(action.kind) && (!Number.isFinite(action.amountToBb) || action.amountToBb! <= 0))) throw new Error();
        if (action.nextByRiver !== undefined) {
          if (!turnRoot || board.length !== 4 || action.nextNodeId !== null || action.nextByRiver.length !== 48
            || new Set(action.nextByRiver.map(([card]) => card)).size !== 48
            || action.nextByRiver.some(([card,id]) => !Number.isInteger(card) || card < 0 || card > 51 || board.includes(card)
              || JSON.stringify(nodes.get(id)?.board) !== JSON.stringify([...board,card]))) throw new Error();
        }
      }
      const keys = new Set(node.rows.map((r) => r.combo));
      const legalRange = spot.ranges[node.actor].filter(([key]) => !spotCombo(key).some((c) => board.includes(c)));
      if (keys.size !== node.rows.length || node.rows.length !== legalRange.length
        || legalRange.some(([key]) => !keys.has(key))) throw new Error();
      for (const row of node.rows) {
        if (row.probabilities.length !== node.actions.length
          || row.probabilities.some((p) => !Number.isFinite(p) || p < 0 || p > 1)
          || !near(row.probabilities.reduce((s,p) => s+p,0),1)
          || (row.evBb !== null && (row.evBb.length !== node.actions.length || row.evBb.some((e) => !Number.isFinite(e) || Math.abs(e) > input.depthBb)))) throw new Error();
        // Rust serializes f32 using its shortest round-trip decimal. JS parses
        // JSON as f64: round back to the actual verified storage precision.
        row.probabilities = row.probabilities.map(Math.fround);
        const storedSum = row.probabilities.reduce((sum,p) => sum+p,0);
        row.probabilities = row.probabilities.map((p) => p/storedSum);
        row.evBb = row.evBb?.map(Math.fround) ?? null;
      }
    }
    return spot;
  } catch { throw new Error('The saved spot is invalid or has not passed verification. No fallback strategy will be used.'); }
}

export async function loadSolvedSpot(entry: SpotCatalogEntry, signal?: AbortSignal): Promise<SolvedSpot> {
  if (!/^\/models\/solved-spots\/[a-z0-9-]+\/[a-f0-9]{64}\.json$/.test(entry.url)
    || !/^[a-f0-9]{64}$/.test(entry.sha256) || entry.bytes <= 0 || entry.bytes > 5_000_000) throw new Error('Invalid spot catalog');
  const response = await fetch(entry.url, { signal, cache: 'force-cache' });
  if (!response.ok) throw new Error('The saved spot could not be loaded. Please retry.');
  const bytes = await response.arrayBuffer();
  const sha = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (b) => b.toString(16).padStart(2,'0')).join('');
  if (bytes.byteLength !== entry.bytes || sha !== entry.sha256) throw new Error('Saved spot checksum mismatch. No decision was graded.');
  const spot = validateSolvedSpot(JSON.parse(new TextDecoder().decode(bytes)));
  if (spot.input.id !== entry.id || spot.input.depthBb !== entry.depthBb || JSON.stringify(spot.input.board) !== JSON.stringify(entry.board)) throw new Error('Saved spot/catalog mismatch');
  return spot;
}

function randomUnit(random: () => number): number {
  const value = random();
  if (!Number.isFinite(value) || value < 0 || value >= 1) throw new Error('Invalid RNG value');
  return value;
}

/** Sample the joint range with exact card removal, not two independent hands. */
export function startSpot(spot: SolvedSpot, version: string, hero: Seat, random = Math.random): SpotSession {
  const deals: Array<{ keys: [number,number]; weight: number }> = [];
  for (const [a, wa] of spot.ranges[0]) for (const [b, wb] of spot.ranges[1]) {
    if (compatible(spotCombo(a), spotCombo(b))) deals.push({ keys: [a,b], weight: wa * wb });
  }
  const total = deals.reduce((sum, d) => sum + d.weight, 0);
  if (!Number.isFinite(total) || total <= 0) throw new Error('No compatible private cards');
  let roll = randomUnit(random) * total;
  const deal = deals.find((d) => (roll -= d.weight) < 0) ?? deals.at(-1)!;
  const cards = deal.keys.map(spotCombo);
  const known = [...spot.input.board, ...cards.flat()];
  const deck = Array.from({length:52}, (_,i) => i).filter((c) => !known.includes(c));
  // A turn strategy must never depend on this shuffled, unrevealed runout.
  if (spot.input.board.length === 4) for (let i=deck.length-1; i>0; i--) {
    const j = Math.floor(randomUnit(random) * (i+1));
    [deck[i],deck[j]] = [deck[j],deck[i]];
  }
  const committed = spot.input.potBb / 2;
  const state: HandState = {
    id: `spot-${spot.input.id}-${Date.now()}-${randomUnit(random).toString(36).slice(2)}`,
    modelVersion: version, depthBb: spot.input.depthBb, button: 'button-small-blind', hero,
    street: spot.input.board.length === 4 ? 'turn' : 'river', holeCards: { 'button-small-blind': cards[0], 'big-blind': cards[1] },
    board: [...spot.input.board], deck,
    potBb: spot.input.potBb, stacksBb: { 'button-small-blind': spot.input.depthBb - committed, 'big-blind': spot.input.depthBb - committed },
    streetBetsBb: { 'button-small-blind': 0, 'big-blind': 0 }, totalCommittedBb: { 'button-small-blind': committed, 'big-blind': committed },
    toAct: 'big-blind', pendingActors: ['big-blind','button-small-blind'], lastFullRaiseBb: 1, raiseReopened: true,
    actionHistory: [], terminal: false, result: null,
  };
  return { state, nodeId: spot.rootNodeId };
}

export function sessionNode(spot: SolvedSpot, session: SpotSession): SolvedSpotNode {
  const { state } = session;
  const node = spot.nodes.find((n) => n.id === session.nodeId);
  const expectedBoard = node?.board ?? spot.input.board;
  if (!node || state.terminal || state.street !== (expectedBoard.length === 4 ? 'turn' : 'river') || state.depthBb !== spot.input.depthBb
    || JSON.stringify(state.board) !== JSON.stringify(expectedBoard) || state.toAct !== SPOT_SEATS[node.actor]
    || SPOT_SEATS.some((seat, p) => !near(state.totalCommittedBb[seat],node.investedBb[p])
      || !near(state.streetBetsBb[seat],node.streetBetsBb[p]) || !near(state.stacksBb[seat],state.depthBb - node.investedBb[p]))) {
    throw new Error('This hand is outside the saved solution. No fallback strategy will be used.');
  }
  return node;
}

export function spotActions(spot: SolvedSpot, session: SpotSession): PolicyAction[] {
  const node = sessionNode(spot, session);
  const cards = session.state.holeCards[SPOT_SEATS[node.actor]];
  const row = node.rows.find((r) => r.combo === comboKey(...cards));
  if (!row) throw new Error('Private hand is missing from the saved solution');
  return node.actions.map((a,i) => ({ ...a, probability: row.probabilities[i],
    evBb: row.evBb?.[i] ?? null, standardErrorBb: row.evBb === null ? null : 0,
    confidence: row.evBb === null ? 'unavailable' : 'high' }));
}

export function advanceSpot(spot: SolvedSpot, session: SpotSession, actionId: string): SpotSession {
  const node = sessionNode(spot, session);
  const action = node.actions.find((a) => a.id === actionId);
  if (!action) throw new Error('Action is not part of this saved solution');
  const state = applyAction(session.state, action);
  let nodeId = action.nextNodeId;
  if (action.nextByRiver) {
    if (state.terminal || state.board.length !== 5) throw new Error('Saved river transition disagrees with the hand engine');
    const edge = action.nextByRiver.find(([card]) => card === state.board[4]);
    if (!edge) throw new Error('The dealt river is missing from this saved solution');
    nodeId = edge[1];
  }
  const next = { state, nodeId };
  if (next.state.terminal !== (next.nodeId === null)) throw new Error('Saved solution/hand engine disagree on settlement');
  if (!next.state.terminal) sessionNode(spot, next);
  return next;
}

export function playSpotOpponent(spot: SolvedSpot, session: SpotSession, random = Math.random): SpotSession {
  let current = session;
  // BB can call a BTN turn bet and then act first again on the river.
  // Keep those two public actions in the log; never show opponent controls.
  for (let step=0; step<16; step++) {
    if (current.state.terminal || current.state.toAct === current.state.hero) return current;
    const actions = spotActions(spot, current);
    let roll = randomUnit(random) * actions.reduce((s,a) => s+a.probability,0);
    const action = actions.find((a) => (roll -= a.probability) < 0) ?? actions.at(-1)!;
    current = advanceSpot(spot, current, action.id);
  }
  throw new Error('Saved opponent line exceeds the bounded game');
}

export function spotEvLoss(actions: PolicyAction[], chosenId: string): number | null {
  const chosen = actions.find((a) => a.id === chosenId);
  if (!chosen || chosen.evBb === null || actions.some((a) => a.evBb === null)) return null;
  return Math.max(0, ...actions.map((a) => a.evBb! - chosen.evBb!));
}
