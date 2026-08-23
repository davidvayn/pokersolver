import { describe, expect, it, vi } from 'vitest';
import { applyAction, createHand, seededRandom } from '@/lib/practice-engine';
import { PracticeContinuationCache } from '@/lib/practice-continuation';

describe('PracticeContinuationCache', () => {
  it('starts one solve for the same exact continuation', async () => {
    const state = createHand({
      id: 'continuation-dedupe',
      modelVersion: 'test-v1',
      depthBb: 20,
      button: 'button-small-blind',
      hero: 'button-small-blind',
      random: seededRandom(7),
    });
    const cache = new PracticeContinuationCache<string>();
    const load = vi.fn(async () => 'prepared');

    const first = cache.prepare(state, load);
    const second = cache.prepare(state, load);

    expect(second).toBe(first);
    await expect(first.promise).resolves.toBe('prepared');
    expect(load).toHaveBeenCalledTimes(1);
    expect(cache.get(state)).toBe(first);
  });

  it('does not reuse a solve across different action histories', async () => {
    const state = createHand({
      id: 'continuation-branch',
      modelVersion: 'test-v1',
      depthBb: 20,
      button: 'button-small-blind',
      hero: 'button-small-blind',
      random: seededRandom(11),
    });
    const called = applyAction(state, {
      id: 'call',
      kind: 'call',
      label: 'Call 0.5bb',
      amountBb: 0.5,
    });
    const raised = applyAction(state, {
      id: 'raise-2',
      kind: 'raise',
      label: 'Raise to 2bb',
      amountBb: 1.5,
      amountToBb: 2,
    });
    const cache = new PracticeContinuationCache<string>();
    const loadCall = vi.fn(async () => 'call');
    const loadRaise = vi.fn(async () => 'raise');

    await expect(cache.prepare(called, loadCall).promise).resolves.toBe('call');
    await expect(cache.prepare(raised, loadRaise).promise).resolves.toBe('raise');

    expect(loadCall).toHaveBeenCalledTimes(1);
    expect(loadRaise).toHaveBeenCalledTimes(1);
  });

  it('evicts a failed preparation so Retry can solve it again', async () => {
    const state = createHand({
      id: 'continuation-retry',
      modelVersion: 'test-v1',
      depthBb: 20,
      button: 'button-small-blind',
      hero: 'button-small-blind',
      random: seededRandom(13),
    });
    const cache = new PracticeContinuationCache<string>();
    const failure = cache.prepare(state, async () => {
      throw new Error('resolver unavailable');
    });

    await expect(failure.promise).rejects.toThrow('resolver unavailable');
    await Promise.resolve();

    const retry = vi.fn(async () => 'recovered');
    await expect(cache.prepare(state, retry).promise).resolves.toBe('recovered');
    expect(retry).toHaveBeenCalledTimes(1);
  });

  it('replays current progress and streams later progress to a subscriber', async () => {
    const state = createHand({
      id: 'continuation-progress',
      modelVersion: 'test-v1',
      depthBb: 20,
      button: 'button-small-blind',
      hero: 'button-small-blind',
      random: seededRandom(17),
    });
    const cache = new PracticeContinuationCache<string, string>();
    let release: (() => void) | undefined;
    const prepared = cache.prepare(state, async (report) => {
      report('flop dealt');
      await new Promise<void>((resolve) => {
        release = resolve;
      });
      report('policy ready');
      return 'complete';
    });
    await Promise.resolve();
    const observed: string[] = [];
    const unsubscribe = prepared.subscribe((progress) => observed.push(progress));

    expect(prepared.latestProgress).toBe('flop dealt');
    expect(observed).toEqual(['flop dealt']);
    release?.();
    await expect(prepared.promise).resolves.toBe('complete');
    expect(observed).toEqual(['flop dealt', 'policy ready']);

    unsubscribe();
  });
});
