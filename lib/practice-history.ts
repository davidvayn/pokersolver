'use client';

import { OPPONENT_PROFILE_FEATURE_COUNT } from '@/lib/opponent-model';
import { historicalGameIdentity, isPracticeGameIdentity, upgradeHistoryGameIdentity } from '@/lib/practice-game-identity';
import { identityForRules } from '@/lib/practice-game-identity';
import { NL25_STUDY_RULES, requireMoneyUnits } from '@/lib/cash-game-rules';
import { cashUnitsFromBb } from '@/lib/practice-cash';
import { settleCashHand } from '@/lib/cash-settlement';
import type {
  OpponentModelSnapshot,
  OpponentPolicyTrace,
  PracticeHandRecord,
} from '@/lib/practice-types';

export const PRACTICE_DB_NAME = 'poker-lab-practice-v3';
export const PRACTICE_DB_VERSION = 2;
export const PRACTICE_HAND_STORE = 'hands';
const HISTORY_EVENT = 'poker-lab-practice-history-v3';
const HISTORY_CHANNEL = 'poker-lab-practice-history-sync-v3';
const MAX_HANDS = 5_000;

function notifyPracticeHistoryChanged(): void {
  window.dispatchEvent(new Event(HISTORY_EVENT));
  if (typeof window.BroadcastChannel === 'undefined') return;
  const channel = new window.BroadcastChannel(HISTORY_CHANNEL);
  channel.postMessage(HISTORY_EVENT);
  channel.close();
}

function requestResult<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('IndexedDB request failed'));
  });
}

function transactionDone(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () =>
      reject(transaction.error ?? new Error('IndexedDB transaction failed'));
    transaction.onabort = () =>
      reject(transaction.error ?? new Error('IndexedDB transaction aborted'));
  });
}

export function openPracticeDatabase(): Promise<IDBDatabase> {
  if (typeof indexedDB === 'undefined') {
    return Promise.reject(new Error('IndexedDB is unavailable'));
  }
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(PRACTICE_DB_NAME, PRACTICE_DB_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(PRACTICE_HAND_STORE)) {
        const store = database.createObjectStore(PRACTICE_HAND_STORE, {
          keyPath: 'id',
        });
        store.createIndex('completedAt', 'completedAt');
        store.createIndex('modelVersion', 'modelVersion');
        store.createIndex('mode', 'mode');
      }
      const store = request.transaction?.objectStore(PRACTICE_HAND_STORE);
      if (store) {
        if (!store.indexNames.contains('rulesSha256')) store.createIndex('rulesSha256', 'gameIdentity.rulesSha256');
        const cursor = store.openCursor();
        cursor.onsuccess = () => {
          const current = cursor.result;
          if (!current) return;
          if (isPracticeHandRecord(current.value)) current.update(upgradeHistoryGameIdentity(current.value));
          current.continue();
        };
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(request.error ?? new Error('Could not open practice history'));
    request.onblocked = () => reject(new Error('Practice history upgrade is blocked'));
  });
}

export function isPracticeHandRecord(value: unknown): value is PracticeHandRecord {
  if (!value || typeof value !== 'object') return false;
  const hand = value as Partial<PracticeHandRecord>;
  return (
    typeof hand.id === 'string' &&
    hand.id.length > 0 &&
    typeof hand.startedAt === 'number' &&
    Number.isFinite(hand.startedAt) &&
    typeof hand.completedAt === 'number' &&
    Number.isFinite(hand.completedAt) &&
    hand.completedAt >= hand.startedAt &&
    typeof hand.modelVersion === 'string' &&
    (hand.gameIdentity === undefined || isPracticeGameIdentity(hand.gameIdentity)) &&
    typeof hand.depthBb === 'number' &&
    hand.depthBb > 1 &&
    (hand.button === 'button-small-blind' || hand.button === 'big-blind') &&
    (hand.hero === 'button-small-blind' || hand.hero === 'big-blind') &&
    Array.isArray(hand.heroCards) &&
    hand.heroCards.length === 2 &&
    Array.isArray(hand.opponentCards) &&
    hand.opponentCards.length === 2 &&
    Array.isArray(hand.board) &&
    Array.isArray(hand.actions) &&
    Array.isArray(hand.decisions) &&
    hand.decisions.every((decision) => decision && typeof decision === 'object' &&
      (decision.gameIdentity === undefined || isPracticeGameIdentity(decision.gameIdentity) &&
        decision.gameIdentity.rulesSha256 === historicalGameIdentity(hand as PracticeHandRecord).rulesSha256)) &&
    validRecordedCashLedger(hand as PracticeHandRecord) &&
    (hand.opponentModel === undefined ||
      isOpponentModelSnapshot(hand.opponentModel)) &&
    (hand.opponentPolicyQueries === undefined ||
      (Array.isArray(hand.opponentPolicyQueries) &&
        hand.opponentPolicyQueries.every(isOpponentPolicyTrace))) &&
    !!hand.result &&
    typeof hand.result === 'object'
  );
}

function validRecordedCashLedger(hand: PracticeHandRecord): boolean {
  const identity = historicalGameIdentity(hand);
  if (!hand.cashLedger) return identity.profileId !== NL25_STUDY_RULES.id;
  try {
    const cash = hand.cashLedger;
    if (identity.rulesSha256 !== identityForRules(cash.rules).rulesSha256
        || hand.button !== 'button-small-blind') return false;
    const seats = ['button-small-blind', 'big-blind'] as const;
    const depth = cashUnitsFromBb(hand.depthBb, cash.rules);
    const vectors = [cash.stacksUnits, cash.streetBetsUnits, cash.committedUnits];
    for (const vector of vectors) {
      if (Object.keys(vector).length !== 2 || !seats.every((seat) => Object.hasOwn(vector, seat))) return false;
      seats.forEach((seat) => requireMoneyUnits(vector[seat]));
    }
    requireMoneyUnits(cash.potUnits); requireMoneyUnits(cash.houseRakeUnits);
    if (seats.reduce((sum,seat) => sum + cash.stacksUnits[seat] + cash.streetBetsUnits[seat], cash.potUnits + cash.houseRakeUnits) !== 2 * depth) return false;
    if (hand.result.reason === 'preflop-complete' || hand.result.reason === 'review-complete') {
      return cash.houseRakeUnits === 0 && hand.result.cashSettlement === undefined;
    }
    if (!hand.result.winner || !hand.result.cashSettlement) return false;
    const expected = settleCashHand(cash.rules, {
      committedUnits: [cash.committedUnits[seats[0]], cash.committedUnits[seats[1]]],
      button: 0, reason: hand.result.reason,
      outcome: hand.result.winner === 'split' ? 'split' : hand.result.winner === seats[0] ? 'player-zero' : 'player-one',
      boardCardsDealt: hand.board.length as 0 | 3 | 4 | 5,
    });
    if (cash.potUnits !== 0 || cash.houseRakeUnits !== expected.rakeUnits || seats.some((seat) => cash.streetBetsUnits[seat] !== 0)) return false;
    for (const key of ['grossPotUnits','rakeUnits','netPotUnits'] as const) {
      if (hand.result.cashSettlement[key] !== expected[key]) return false;
    }
    for (const key of ['uncalledRefundUnits','awardsUnits','netPayoffUnits'] as const) {
      if (!expected[key].every((value,index) => hand.result.cashSettlement?.[key][index] === value)) return false;
    }
    return seats.every((seat,index) => cash.stacksUnits[seat] - depth === expected.netPayoffUnits[index]
      && hand.result.netBb[seat] === expected.netPayoffUnits[index] / cash.rules.unitsPerBb);
  } catch { return false; }
}

function isOpponentPolicyTrace(value: unknown): value is OpponentPolicyTrace {
  if (!value || typeof value !== 'object') return false;
  const trace = value as Partial<OpponentPolicyTrace>;
  const distributions = [
    trace.baselineActions,
    trace.responseActions,
    trace.servedActions,
  ];
  return (
    typeof trace.stateHash === 'string' &&
    /^[a-f0-9]{64}$/.test(trace.stateHash) &&
    typeof trace.modelVersion === 'string' &&
    typeof trace.profileVersion === 'string' &&
    typeof trace.evidenceCount === 'number' &&
    Number.isInteger(trace.evidenceCount) &&
    trace.evidenceCount >= 0 &&
    typeof trace.confidence === 'number' &&
    trace.confidence >= 0 &&
    trace.confidence <= 1 &&
    typeof trace.responseWeight === 'number' &&
    trace.responseWeight >= 0 &&
    trace.responseWeight <= 1 &&
    distributions.every(
      (distribution) =>
        Array.isArray(distribution) &&
        distribution.length > 0 &&
        distribution.every(
          (action) =>
            typeof action.id === 'string' &&
            typeof action.probability === 'number' &&
            Number.isFinite(action.probability) &&
            action.probability >= 0 &&
            action.probability <= 1
        ) &&
        Math.abs(
          distribution.reduce((sum, action) => sum + action.probability, 0) - 1
        ) <= 1e-6
    )
  );
}

function isOpponentModelSnapshot(value: unknown): value is OpponentModelSnapshot {
  if (!value || typeof value !== 'object') return false;
  const profile = value as Partial<OpponentModelSnapshot>;
  return (
    profile.schema === 'local-opponent-profile-v1' &&
    typeof profile.version === 'string' &&
    profile.source === 'local-indexeddb' &&
    typeof profile.observations === 'number' &&
    Number.isInteger(profile.observations) &&
    profile.observations >= 0 &&
    typeof profile.stableEvidence === 'number' &&
    Number.isInteger(profile.stableEvidence) &&
    typeof profile.confidence === 'number' &&
    profile.confidence >= 0 &&
    profile.confidence <= 1 &&
    typeof profile.responseWeight === 'number' &&
    profile.responseWeight >= 0 &&
    profile.responseWeight <= 1 &&
    Array.isArray(profile.features) &&
    profile.features.length === OPPONENT_PROFILE_FEATURE_COUNT &&
    profile.features.every(
      (feature) => typeof feature === 'number' && Number.isFinite(feature)
    )
  );
}

export async function loadPracticeHands(
  limit = MAX_HANDS
): Promise<PracticeHandRecord[]> {
  let database: IDBDatabase | null = null;
  try {
    database = await openPracticeDatabase();
    const transaction = database.transaction(PRACTICE_HAND_STORE, 'readonly');
    const values = await requestResult(
      transaction.objectStore(PRACTICE_HAND_STORE).getAll()
    );
    await transactionDone(transaction);
    return (values as unknown[])
      .filter(isPracticeHandRecord)
      .sort((first, second) => second.completedAt - first.completedAt)
      .slice(0, Math.max(0, limit));
  } catch {
    return [];
  } finally {
    database?.close();
  }
}

async function trimHistory(database: IDBDatabase): Promise<void> {
  const read = database.transaction(PRACTICE_HAND_STORE, 'readonly');
  const keys = await requestResult(
    read.objectStore(PRACTICE_HAND_STORE).index('completedAt').getAllKeys()
  );
  await transactionDone(read);
  const excess = keys.length - MAX_HANDS;
  if (excess <= 0) return;
  const write = database.transaction(PRACTICE_HAND_STORE, 'readwrite');
  const store = write.objectStore(PRACTICE_HAND_STORE);
  for (const key of keys.slice(0, excess)) store.delete(key);
  await transactionDone(write);
}

export async function savePracticeHand(
  hand: PracticeHandRecord
): Promise<boolean> {
  if (!isPracticeHandRecord(hand)) return false;
  let database: IDBDatabase | null = null;
  try {
    database = await openPracticeDatabase();
    const transaction = database.transaction(PRACTICE_HAND_STORE, 'readwrite');
    transaction.objectStore(PRACTICE_HAND_STORE).put(upgradeHistoryGameIdentity(hand));
    await transactionDone(transaction);
    await trimHistory(database);
    notifyPracticeHistoryChanged();
    return true;
  } catch {
    return false;
  } finally {
    database?.close();
  }
}

export async function clearPracticeHistory(): Promise<boolean> {
  let database: IDBDatabase | null = null;
  try {
    database = await openPracticeDatabase();
    const transaction = database.transaction(PRACTICE_HAND_STORE, 'readwrite');
    transaction.objectStore(PRACTICE_HAND_STORE).clear();
    await transactionDone(transaction);
    notifyPracticeHistoryChanged();
    return true;
  } catch {
    return false;
  } finally {
    database?.close();
  }
}

export function subscribePracticeHistory(listener: () => void): () => void {
  if (typeof window === 'undefined') return () => undefined;
  window.addEventListener(HISTORY_EVENT, listener);
  const channel =
    typeof window.BroadcastChannel === 'undefined'
      ? null
      : new window.BroadcastChannel(HISTORY_CHANNEL);
  channel?.addEventListener('message', listener);
  return () => {
    window.removeEventListener(HISTORY_EVENT, listener);
    channel?.removeEventListener('message', listener);
    channel?.close();
  };
}
