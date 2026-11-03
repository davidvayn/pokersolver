import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import flopCatalog from '@/data/solved-spots/flop-catalog.json';
import catalog from '@/data/solved-spots/catalog.json';
import { prepareSpotLine } from '@/lib/solved-spot-line';
import { beginSpotAction, loadSolvedSpot, spotActions, startSpot, type SpotCatalogEntry, type SpotSession } from '@/lib/solved-spots';
import { seededRandom } from '@/lib/practice-engine';

const entry = flopCatalog[0] as SpotCatalogEntry;
function fixtureFetch(fail: () => boolean = () => false) {
  return vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) =>
    fail() ? new Response('', { status: 503 }) : new Response(readFileSync(path.join(process.cwd(), 'public', String(url)))));
}

describe('asynchronous saved-spot line', () => {
  it('publishes the opponent action before a failed shard, then retries without resampling or reapplying it', async () => {
    let fail = false;
    const fetcher = fixtureFetch(() => { const result = fail; fail = false; return result; });
    try {
      const spot = await loadSolvedSpot(entry);
      const start = startSpot(spot, entry.sha256, 'button-small-blind', seededRandom(17));
      let cursor = start;
      const random = vi.fn(() => 0);
      fail = true;
      await expect(prepareSpotLine(spot, start, 'decision', next => { cursor = next; }, random)).rejects.toThrow('retry');
      expect(cursor.state.id).toBe(start.state.id);
      expect(cursor.state.actionHistory).toHaveLength(1);
      expect(random).toHaveBeenCalledTimes(1);
      const pending = JSON.stringify(cursor);
      const ready = await prepareSpotLine(spot, cursor, 'decision', next => { cursor = next; }, random);
      expect(JSON.stringify(ready)).toBe(pending);
      expect(ready.state.toAct).toBe(ready.state.hero);
      expect(random).toHaveBeenCalledTimes(1);
    } finally { fetcher.mockRestore(); }
  });

  it('stops for user feedback before taking any opponent action', async () => {
    const fetcher = fixtureFetch();
    try {
      const spot = await loadSolvedSpot(entry);
      const start = startSpot(spot, entry.sha256, 'big-blind', seededRandom(22));
      const check = spotActions(spot, start).find(a => a.kind === 'check')!;
      const pending = beginSpotAction(spot, start, check.id);
      const random = vi.fn(() => 0);
      const published: SpotSession[] = [];
      const ready = await prepareSpotLine(spot, pending, 'feedback', next => published.push(next), random);
      expect(published[0]).toBe(pending);
      expect(ready.state.actionHistory).toHaveLength(1);
      expect(ready.state.toAct).not.toBe(ready.state.hero);
      expect(random).not.toHaveBeenCalled();
    } finally { fetcher.mockRestore(); }
  });

  it.each(['big-blind', 'button-small-blind'] as const)('plays complete saved flop hands from %s without online solving', async hero => {
    const fetcher = fixtureFetch();
    try {
      const spot = await loadSolvedSpot(entry);
      let current = await prepareSpotLine(spot, startSpot(spot, entry.sha256, hero, seededRandom(22)), 'decision', () => {}, () => 0);
      const streets = new Set<string>();
      for (let decision = 0; decision < 12 && !current.state.terminal; decision++) {
        streets.add(current.state.street);
        const action = spotActions(spot, current).find(a => a.kind === 'check' || a.kind === 'call')!;
        expect(action).toBeDefined();
        current = await prepareSpotLine(spot, beginSpotAction(spot, current, action.id), 'feedback', () => {});
        current = await prepareSpotLine(spot, current, 'decision', () => {}, () => 0);
      }
      expect([...streets]).toEqual(['flop', 'turn', 'river']);
      expect(current.state.result?.reason).toBe('showdown');
      expect(current.state.modelVersion).toBe(entry.sha256);
      expect(fetcher.mock.calls.every(([url]) => String(url).startsWith('/models/solved-spots/'))).toBe(true);
    } finally { fetcher.mockRestore(); }
  });

  it('retains the original synchronous river corpus behavior', async () => {
    const fetcher = fixtureFetch();
    try {
      const spot = await loadSolvedSpot(catalog[0] as SpotCatalogEntry);
      const start = startSpot(spot, catalog[0].sha256, 'big-blind', seededRandom(5));
      const action = spotActions(spot, start).find(a => a.kind === 'check')!;
      const ready = await prepareSpotLine(spot, beginSpotAction(spot, start, action.id), 'feedback', () => {});
      expect(ready.state.actionHistory).toHaveLength(1);
      expect(ready.state.street).toBe('river');
    } finally { fetcher.mockRestore(); }
  });
});
