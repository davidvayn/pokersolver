import { comboKey, type Combo } from '@/lib/cards';
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
}
export interface SpotCatalogEntry {
  id: string; title: string; board: number[]; depthBb: number;
  url: string; sha256: string; bytes: number; verification: SpotVerification;
}
export interface SolvedSpot {
  schema: 'solved-river-spot-v1';
  inputSha256: string;
  input: {
    id: string; title: string; description: string; rangeAssumptions: string;
    board: number[]; depthBb: number; potBb: number;
    rangeClasses: [string[], string[]]; betPotFractions: number[];
    raisePotFractions: number[]; raiseCap: number; iterations: number;
  };
  rootNodeId: string;
  ranges: [Array<[number, number]>, Array<[number, number]>];
  nodes: SolvedSpotNode[];
  verification: SpotVerification;
}
export interface SolvedSpotNode {
  id: string; actor: 0 | 1; investedBb: [number, number]; streetBetsBb: [number, number];
  actions: Array<LegalAction & { nextNodeId: string | null }>;
  rows: Array<{ combo: number; probabilities: number[]; evBb: number[] | null }>;
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

/** Structural checks in addition to the content hash and offline exact checker. */
export function validateSolvedSpot(value: unknown): SolvedSpot {
  const spot = value as SolvedSpot;
  try {
    const v = spot.verification;
    if (spot.schema !== 'solved-river-spot-v1' || !/^[a-f0-9]{64}$/.test(spot.inputSha256)
      || !v.accepted || v.method !== 'direct-compatible-pair-enumeration-of-exported-f32-policy-v1'
      || !Number.isFinite(v.totalBestResponseGainBb) || v.totalBestResponseGainBb < 0 || v.totalBestResponseGainBb > .05
      || !Number.isFinite(v.maximumActionEvErrorBb) || v.maximumActionEvErrorBb < 0 || v.maximumActionEvErrorBb > .0001
      || !Number.isFinite(v.maximumProbabilitySumError) || v.maximumProbabilitySumError > 1e-6
      || !Number.isFinite(v.zeroSumResidualBb) || v.zeroSumResidualBb > 1e-5) throw new Error();
    const input = spot.input;
    if (!Number.isFinite(input.depthBb) || input.depthBb < 2 || input.depthBb > 1000
      || !Number.isFinite(input.potBb) || input.potBb < 2 || input.potBb >= input.depthBb * 2
      || input.board.length !== 5 || new Set(input.board).size !== 5
      || input.board.some((c) => !Number.isInteger(c) || c < 0 || c > 51)
      || spot.ranges.length !== 2 || spot.nodes.length < 1 || spot.nodes.length > 1000
      || !Number.isInteger(input.raiseCap) || input.raiseCap < 0 || input.raiseCap > 1
      || [input.betPotFractions,input.raisePotFractions].some((sizes) => !sizes.length || sizes.length > 2
        || sizes.some((f,i) => !Number.isFinite(f) || f <= 0 || f > 4 || (i>0 && f <= sizes[i-1])))) throw new Error();
    for (const range of spot.ranges) {
      if (!range.length || new Set(range.map(([key]) => key)).size !== range.length
        || range.some(([key, weight]) => !Number.isFinite(weight) || weight <= 0 || weight > 1
          || spotCombo(key).some((card) => input.board.includes(card)))
        || !near(range.reduce((sum, [,w]) => sum + w, 0), 1)) throw new Error();
    }
    const ids = new Set(spot.nodes.map((node) => node.id));
    if (ids.size !== spot.nodes.length || !ids.has(spot.rootNodeId)) throw new Error();
    for (const node of spot.nodes) {
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
      }
      const keys = new Set(node.rows.map((r) => r.combo));
      if (keys.size !== node.rows.length || node.rows.length !== spot.ranges[node.actor].length
        || spot.ranges[node.actor].some(([key]) => !keys.has(key))) throw new Error();
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
  const committed = spot.input.potBb / 2;
  const state: HandState = {
    id: `spot-${spot.input.id}-${Date.now()}-${randomUnit(random).toString(36).slice(2)}`,
    modelVersion: version, depthBb: spot.input.depthBb, button: 'button-small-blind', hero,
    street: 'river', holeCards: { 'button-small-blind': cards[0], 'big-blind': cards[1] },
    board: [...spot.input.board], deck: Array.from({length:52}, (_,i) => i).filter((c) => !known.includes(c)),
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
  if (!node || state.terminal || state.street !== 'river' || state.depthBb !== spot.input.depthBb
    || JSON.stringify(state.board) !== JSON.stringify(spot.input.board) || state.toAct !== SPOT_SEATS[node.actor]
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
  const next = { state: applyAction(session.state, action), nodeId: action.nextNodeId };
  if (next.state.terminal !== (next.nodeId === null)) throw new Error('Saved solution/hand engine disagree on settlement');
  if (!next.state.terminal) sessionNode(spot, next);
  return next;
}

export function playSpotOpponent(spot: SolvedSpot, session: SpotSession, random = Math.random): SpotSession {
  if (session.state.terminal || session.state.toAct === session.state.hero) return session;
  const actions = spotActions(spot, session);
  let roll = randomUnit(random) * actions.reduce((s,a) => s+a.probability,0);
  const action = actions.find((a) => (roll -= a.probability) < 0) ?? actions.at(-1)!;
  return advanceSpot(spot, session, action.id);
}

export function spotEvLoss(actions: PolicyAction[], chosenId: string): number | null {
  const chosen = actions.find((a) => a.id === chosenId);
  if (!chosen || chosen.evBb === null || actions.some((a) => a.evBb === null)) return null;
  return Math.max(0, ...actions.map((a) => a.evBb! - chosen.evBb!));
}
