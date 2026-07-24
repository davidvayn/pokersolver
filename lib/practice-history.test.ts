import { afterEach, describe, expect, it, vi } from 'vitest';
import { CHARTS } from '@/data/preflop/ranges';
import {
  appendPracticeRecords,
  loadPracticeHistory,
} from '@/lib/practice-history';
import {
  createPracticeQuestion,
  recordPracticeAnswer,
} from '@/lib/practice';

const STORAGE_KEY = 'poker-lab-practice-history-v1';

function installStorage(initial?: unknown, failWrites = false) {
  const values = new Map<string, string>();
  if (initial !== undefined) {
    values.set(STORAGE_KEY, JSON.stringify(initial));
  }
  const storage = {
    getItem: vi.fn((key: string) => values.get(key) ?? null),
    setItem: vi.fn((key: string, value: string) => {
      if (failWrites) throw new Error('quota');
      values.set(key, value);
    }),
    removeItem: vi.fn((key: string) => values.delete(key)),
  };
  vi.stubGlobal('localStorage', storage);
  vi.stubGlobal('window', {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  });
  return storage;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('practice history', () => {
  it('drops malformed records and migrates legacy arrays', () => {
    const chart = CHARTS.find((candidate) => candidate.id === 'rfi-BTN')!;
    const valid = recordPracticeAnswer(
      createPracticeQuestion(chart, 'AA', 6, 'valid'),
      'Raise',
      100,
      Date.UTC(2026, 6, 23)
    );
    installStorage([valid, { id: 'broken', answeredAt: Number.NaN }]);

    expect(loadPracticeHistory()).toEqual([valid]);
  });

  it('keeps the session usable when storage writes fail', () => {
    installStorage([], true);
    const chart = CHARTS.find((candidate) => candidate.id === 'rfi-BTN')!;
    const record = recordPracticeAnswer(
      createPracticeQuestion(chart, 'AA', 6, 'valid'),
      'Raise',
      100
    );

    expect(appendPracticeRecords([record])).toBe(false);
  });
});
