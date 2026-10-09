import 'fake-indexeddb/auto';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  clearPracticeHistory,
  loadPracticeHands,
  PRACTICE_DB_NAME,
  savePracticeHand,
  subscribePracticeHistory,
} from '@/lib/practice-history';
import { analyzePractice } from '@/lib/practice-stats';
import { buildOpponentModel } from '@/lib/opponent-model';
import { HOME_GAME_IDENTITY, HOME_RULES_SHA256, identityForRules } from '@/lib/practice-game-identity';
import { NL25_STUDY_RULES } from '@/lib/cash-game-rules';
import { applyAction, createHand, seededRandom } from '@/lib/practice-engine';
import type {
  PracticeDecisionRecord,
  PracticeHandRecord,
} from '@/lib/practice-types';

function decision(id: string, loss: number | null, lowConfidence = false): PracticeDecisionRecord {
  return {
    id,
    handId: `hand-${id}`,
    answeredAt: Number(id.replace(/\D/g, '')) + 1,
    responseMs: 800,
    modelVersion: 'test-v1',
    mode: 'full-hand',
    depthBb: 20,
    street: 'flop',
    position: 'button-small-blind',
    handBucket: 'AKs',
    facingAction: 'check',
    stateHash: 'a'.repeat(64),
    board: [0, 1, 2],
    heroCards: [50, 46],
    chosenAction: { id: 'check', kind: 'check', label: 'Check' },
    policyActions: [],
    chosenActionEvBb: loss === null ? null : 0,
    bestActionEvBb: loss,
    evLossBb: loss,
    // Grades now describe policy frequency, independently of EV availability.
    grade: loss !== null && loss > 0.25 ? 'blunder' : 'good',
    confidence: lowConfidence ? 'low' : loss === null ? 'unavailable' : 'high',
    lowConfidence,
  };
}

function hand(id: string, decisions: PracticeDecisionRecord[]): PracticeHandRecord {
  return {
    id,
    startedAt: 1,
    completedAt: Number(id.replace(/\D/g, '')) + 2,
    modelVersion: 'test-v1',
    mode: 'full-hand',
    depthBb: 20,
    button: 'button-small-blind',
    hero: 'button-small-blind',
    heroCards: [50, 46],
    opponentCards: [0, 1],
    board: [2, 3, 4, 5, 6],
    actions: [],
    decisions,
    result: {
      reason: 'showdown',
      winner: 'button-small-blind',
      potBb: 4,
      netBb: { 'button-small-blind': 2, 'big-blind': -2 },
    },
  };
}

function deleteDatabase(): Promise<void> {
  return new Promise((resolve) => {
    const request = indexedDB.deleteDatabase(PRACTICE_DB_NAME);
    request.onsuccess = () => resolve();
    request.onerror = () => resolve();
    request.onblocked = () => resolve();
  });
}

beforeEach(async () => {
  vi.stubGlobal('window', new EventTarget());
  await deleteDatabase();
});

afterEach(async () => {
  await deleteDatabase();
  vi.unstubAllGlobals();
});

describe('fresh IndexedDB practice history', () => {
  it('upgrades recognized Home history and preserves unresolved records without invented rake', async () => {
    const recognized = { ...hand('h1',[decision('d1',.1)]), modelVersion:'hu-push-fold-v1' };
    const unknown = hand('h2',[decision('d2',.2)]);
    await new Promise<void>((resolve,reject) => {
      const request = indexedDB.open(PRACTICE_DB_NAME,1);
      request.onupgradeneeded = () => {
        const store = request.result.createObjectStore('hands',{ keyPath:'id' });
        store.createIndex('completedAt','completedAt'); store.createIndex('modelVersion','modelVersion'); store.createIndex('mode','mode');
      };
      request.onsuccess = () => {
        const db = request.result; const tx = db.transaction('hands','readwrite');
        tx.objectStore('hands').put(recognized); tx.objectStore('hands').put(unknown);
        tx.oncomplete = () => { db.close(); resolve(); }; tx.onerror = () => reject(tx.error);
      };
      request.onerror = () => reject(request.error);
    });
    const records = await loadPracticeHands();
    expect(records.find((record) => record.id === 'h1')?.gameIdentity).toEqual({ ...HOME_GAME_IDENTITY,source:'known-legacy-home' });
    expect(records.find((record) => record.id === 'h2')?.gameIdentity?.source).toBe('unresolved');
    expect(records.find((record) => record.id === 'h1')?.result).toEqual(recognized.result);
    expect(records.find((record) => record.id === 'h2')?.decisions[0].evLossBb).toBe(.2);
    expect(records.every((record) => record.result.cashSettlement === undefined)).toBe(true);
  });

  it('keeps cash settlement and scoped evidence separate from Home and unknown history', async () => {
    const state = applyAction(createHand({ modelVersion:'cash-pilot',depthBb:20,button:'button-small-blind',
      hero:'button-small-blind',cashRules:NL25_STUDY_RULES,random:seededRandom(7) }), { id:'fold',kind:'fold',label:'Fold' });
    const cash = { ...hand('h3',[]), modelVersion:state.modelVersion,gameIdentity:identityForRules(NL25_STUDY_RULES),
      heroCards:state.holeCards[state.hero],opponentCards:state.holeCards['big-blind'],board:state.board,
      cashLedger:state.cash,result:state.result! };
    expect(await savePracticeHand(cash)).toBe(true);
    expect(await savePracticeHand({ ...cash,cashLedger:undefined })).toBe(false);
    const home = { ...hand('h4',[decision('d4',.1)]),gameIdentity:HOME_GAME_IDENTITY };
    const deeper = { ...home,id:'h5',depthBb:50 };
    const profile = buildOpponentModel([home,cash,deeper,hand('h6',[decision('d6',.2)])],'baseline',undefined,
      { rulesSha256:HOME_RULES_SHA256,depthBb:20 });
    expect(profile.observations).toBe(1);
    expect(analyzePractice([home,cash,deeper],Date.now(),{ rulesSha256:HOME_RULES_SHA256,depthBb:20 }).hands).toBe(1);
    expect((await loadPracticeHands())[0].result.cashSettlement?.rakeUnits).toBe(0);
  });
  it('stores, orders, and clears complete hand records', async () => {
    expect(await savePracticeHand(hand('h1', [decision('d1', 0.1)]))).toBe(true);
    expect(await savePracticeHand(hand('h2', [decision('d2', 0.2)]))).toBe(true);
    expect((await loadPracticeHands()).map((record) => record.id)).toEqual([
      'h2',
      'h1',
    ]);
    expect(await clearPracticeHistory()).toBe(true);
    expect(await loadPracticeHands()).toEqual([]);
  });

  it('notifies live stats subscribers when practice writes new history', async () => {
    const listener = vi.fn();
    const unsubscribe = subscribePracticeHistory(listener);
    expect(await savePracticeHand(hand('h4', [decision('d4', 0.1)]))).toBe(true);
    expect(listener).toHaveBeenCalledOnce();
    unsubscribe();
    expect(await clearPracticeHistory()).toBe(true);
    expect(listener).toHaveBeenCalledOnce();
  });

  it('refreshes subscribers when another tab broadcasts a history change', () => {
    const channels: EventTarget[] = [];
    class TestBroadcastChannel extends EventTarget {
      constructor(_name: string) {
        super();
        channels.push(this);
      }

      close() {}
    }
    Object.assign(window, { BroadcastChannel: TestBroadcastChannel });
    const listener = vi.fn();
    const unsubscribe = subscribePracticeHistory(listener);
    channels[0].dispatchEvent(new Event('message'));
    expect(listener).toHaveBeenCalledOnce();
    unsubscribe();
  });

  it('ignores malformed writes instead of contaminating the fresh schema', async () => {
    expect(await savePracticeHand({ id: 'broken' } as PracticeHandRecord)).toBe(false);
    expect(await loadPracticeHands()).toEqual([]);
  });

  it('retains the local profile and auditable policy blend without a server database', async () => {
    const record = hand('h3', [decision('d3', 0.02)]);
    record.opponentModel = buildOpponentModel([], 'baseline');
    record.opponentPolicyQueries = [
      {
        stateHash: 'b'.repeat(64),
        modelVersion: record.modelVersion,
        profileVersion: record.opponentModel.version,
        evidenceCount: 0,
        confidence: 0,
        responseWeight: 0,
        baselineActions: [{ id: 'check', probability: 1 }],
        responseActions: [{ id: 'check', probability: 1 }],
        servedActions: [{ id: 'check', probability: 1 }],
      },
    ];
    expect(await savePracticeHand(record)).toBe(true);
    const [loaded] = await loadPracticeHands();
    expect(loaded.opponentModel?.source).toBe('local-indexeddb');
    expect(loaded.opponentPolicyQueries?.[0]).toMatchObject({
      responseWeight: 0,
      evidenceCount: 0,
    });
  });

  it('aggregates EV loss, ungraded decisions, confidence, and costly records', () => {
    const records = [
      hand('h1', [decision('d1', 0.4, true), decision('d2', 0.1)]),
      hand('h2', [decision('d3', null)]),
    ];
    const stats = analyzePractice(records);
    expect(stats.hands).toBe(2);
    expect(stats.decisions).toBe(3);
    expect(stats.gradedDecisions).toBe(2);
    expect(stats.averageEvLossBb).toBeCloseTo(0.25);
    expect(stats.totalEvLossBb).toBeCloseTo(0.5);
    expect(stats.lowConfidencePercentage).toBeCloseTo(1 / 3);
    expect(stats.recentCostly[0].id).toBe('d1');
  });
});
