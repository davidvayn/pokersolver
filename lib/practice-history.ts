'use client';

import type { PracticeRecord } from '@/lib/practice-types';
import { POSITION_LABELS } from '@/lib/positions';

const STORAGE_KEY = 'poker-lab-practice-history-v1';
const HISTORY_EVENT = 'poker-lab-practice-history';
const MAX_RECORDS = 5000;
const STORAGE_VERSION = 2;
const ACTIONS = new Set(['Fold', 'Call', 'Raise', '3-bet', 'All-in']);
const CATEGORIES = new Set(['RFI', 'vs-RFI']);
const POSITIONS = new Set(Object.keys(POSITION_LABELS));

interface StoredHistory {
  version: 1 | typeof STORAGE_VERSION;
  records: PracticeRecord[];
}

export function loadPracticeHistory(): PracticeRecord[] {
  if (typeof window === 'undefined') return [];
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]');
    const records =
      Array.isArray(parsed)
        ? parsed
        : isStoredHistory(parsed)
          ? parsed.records
          : [];
    return records.filter(isPracticeRecord).slice(-MAX_RECORDS);
  } catch {
    return [];
  }
}

export function appendPracticeRecords(records: PracticeRecord[]): boolean {
  if (typeof window === 'undefined' || records.length === 0) return false;
  try {
    const merged = new Map(
      [...loadPracticeHistory(), ...records]
        .filter(isPracticeRecord)
        .map((record) => [record.id, record])
    );
    const stored: StoredHistory = {
      version: STORAGE_VERSION,
      records: [...merged.values()]
        .sort((first, second) => first.answeredAt - second.answeredAt)
        .slice(-MAX_RECORDS),
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(stored));
    window.dispatchEvent(new Event(HISTORY_EVENT));
    return true;
  } catch {
    return false;
  }
}

export function clearPracticeHistory(): boolean {
  if (typeof window === 'undefined') return false;
  try {
    localStorage.removeItem(STORAGE_KEY);
    window.dispatchEvent(new Event(HISTORY_EVENT));
    return true;
  } catch {
    return false;
  }
}

export function subscribePracticeHistory(listener: () => void): () => void {
  if (typeof window === 'undefined') return () => undefined;
  window.addEventListener(HISTORY_EVENT, listener);
  window.addEventListener('storage', listener);
  return () => {
    window.removeEventListener(HISTORY_EVENT, listener);
    window.removeEventListener('storage', listener);
  };
}

function isPracticeRecord(value: unknown): value is PracticeRecord {
  if (!value || typeof value !== 'object') return false;
  const record = value as Partial<PracticeRecord>;
  return (
    typeof record.id === 'string' &&
    record.id.length > 0 &&
    typeof record.answeredAt === 'number' &&
    Number.isFinite(record.answeredAt) &&
    record.answeredAt > 0 &&
    typeof record.chartId === 'string' &&
    record.chartId.length > 0 &&
    CATEGORIES.has(record.category ?? '') &&
    (record.seats === 2 || record.seats === 6 || record.seats === 9) &&
    POSITIONS.has(record.hero ?? '') &&
    (record.villain === undefined || POSITIONS.has(record.villain)) &&
    typeof record.handClass === 'string' &&
    /^[2-9TJQKA]{2}(?:[so])?$/.test(record.handClass) &&
    ACTIONS.has(record.chosenAction ?? '') &&
    ACTIONS.has(record.recommendedAction ?? '') &&
    typeof record.correct === 'boolean' &&
    typeof record.responseMs === 'number' &&
    Number.isFinite(record.responseMs) &&
    record.responseMs >= 0 &&
    record.responseMs <= 86_400_000 &&
    (record.scenario === undefined || isScenarioSnapshot(record.scenario))
  );
}

function isStoredHistory(value: unknown): value is StoredHistory {
  if (!value || typeof value !== 'object') return false;
  const stored = value as Partial<StoredHistory>;
  return (
    (stored.version === 1 || stored.version === STORAGE_VERSION) &&
    Array.isArray(stored.records)
  );
}

function isScenarioSnapshot(value: unknown): boolean {
  if (!value || typeof value !== 'object') return false;
  const scenario = value as NonNullable<PracticeRecord['scenario']>;
  if (
    typeof scenario.scenarioId !== 'string' ||
    scenario.scenarioId.length === 0 ||
    typeof scenario.label !== 'string' ||
    !Number.isFinite(scenario.effectiveStackBb) ||
    scenario.effectiveStackBb <= 0 ||
    !scenario.openingSize ||
    typeof scenario.openingSize !== 'object' ||
    !scenario.provenance ||
    typeof scenario.provenance !== 'object'
  ) {
    return false;
  }
  const opening = scenario.openingSize;
  const validOpening =
    opening.kind === 'all-in' ||
    opening.kind === 'unspecified' ||
    (opening.kind === 'raise-to' &&
      Number.isFinite(opening.bb) &&
      opening.bb > 0);
  const provenance = scenario.provenance;
  return (
    validOpening &&
    (provenance.source === 'curated' ||
      provenance.source === 'offline-solver') &&
    (provenance.status === 'reference' ||
      provenance.status === 'validated' ||
      provenance.status === 'approximate') &&
    typeof provenance.model === 'string'
  );
}
