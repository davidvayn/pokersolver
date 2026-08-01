'use client';

import type { PracticeHandRecord } from '@/lib/practice-types';

export const PRACTICE_DB_NAME = 'poker-lab-practice-v3';
export const PRACTICE_DB_VERSION = 1;
export const PRACTICE_HAND_STORE = 'hands';
const HISTORY_EVENT = 'poker-lab-practice-history-v3';
const MAX_HANDS = 5_000;

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
    !!hand.result &&
    typeof hand.result === 'object'
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
    transaction.objectStore(PRACTICE_HAND_STORE).put(hand);
    await transactionDone(transaction);
    await trimHistory(database);
    window.dispatchEvent(new Event(HISTORY_EVENT));
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
    window.dispatchEvent(new Event(HISTORY_EVENT));
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
  return () => window.removeEventListener(HISTORY_EVENT, listener);
}
