import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import catalog from '@/data/solved-spots/catalog.json';
import { comboKey } from '@/lib/cards';
import { SPOT_SEATS, spotRangeSummary, startSpot, validateSolvedSpot, type SolvedSpot } from '@/lib/solved-spots';

const rawSpot = () => JSON.parse(readFileSync(path.join(process.cwd(),'public',catalog[0].url),'utf8')) as SolvedSpot;

describe('declared saved-spot range weights', () => {
  it('labels inclusion multipliers without presenting them as action probabilities', () => {
    const { input } = rawSpot();
    expect(spotRangeSummary(input,0)).toBe(input.rangeClasses[0].join(', '));
    input.rangeClassWeights = input.rangeClasses.map((labels) => labels.map(() => .25)) as [number[],number[]];
    expect(spotRangeSummary(input,0)).toBe(input.rangeClasses[0].map((label) => `${label} ×0.25`).join(', '));
    expect(spotRangeSummary(input,1)).not.toContain('%');
  });
  it('preserves the legacy equal-weight prior and accepts explicitly equivalent weights', () => {
    const original = rawSpot(), explicit = rawSpot();
    expect(original.input.rangeClassWeights).toBeUndefined();
    explicit.input.rangeClassWeights = explicit.input.rangeClasses.map((labels) => labels.map(() => 1)) as [number[],number[]];
    // Structural check only; the native checker also checks input identity.
    expect(validateSolvedSpot(explicit).ranges).toEqual(validateSolvedSpot(original).ranges);
  });

  it('rejects invalid weight shapes, values, duplicate classes and noncanonical labels', () => {
    const cases: unknown[] = [null,[],[[]],[[1],[1]]];
    for (const value of cases) {
      const bad = rawSpot();
      Object.assign(bad.input,{rangeClassWeights:value});
      expect(() => validateSolvedSpot(bad)).toThrow('invalid');
    }
    for (const value of [0,-.1,1.1,NaN,Infinity]) {
      const bad = rawSpot();
      bad.input.rangeClassWeights = bad.input.rangeClasses.map((labels) => labels.map(() => 1)) as [number[],number[]];
      bad.input.rangeClassWeights[0][0] = value;
      expect(() => validateSolvedSpot(bad)).toThrow('invalid');
    }
    for (const labels of [['QQ','QQ'],['AK'],['AAo'],['KAo']]) {
      const bad = rawSpot(); bad.input.rangeClasses[0] = labels;
      expect(() => validateSolvedSpot(bad)).toThrow('invalid');
    }
  });

  it('does not allow a new declared prior to borrow an old frozen range', () => {
    const bad = rawSpot();
    bad.input.rangeClassWeights = bad.input.rangeClasses.map((labels) => labels.map(() => 1)) as [number[],number[]];
    bad.input.rangeClassWeights[0][0] = .5;
    expect(() => validateSolvedSpot(bad)).toThrow('invalid');
    const changed = rawSpot();
    changed.ranges[0][0][1] += .001;
    changed.ranges[0][1][1] -= .001; // Still sums to one, but does not match the declaration.
    expect(() => validateSolvedSpot(changed)).toThrow('invalid');
  });

  it('samples the weighted joint prior conditioned on card compatibility, not independent marginals', () => {
    const aa = comboKey(48,49), kk = comboKey(44,45), jj = comboKey(36,37);
    // A dealing-only fixture, deliberately not certified as a solved policy.
    const fixture = rawSpot(); fixture.nodes = []; fixture.verification.accepted = false;
    fixture.ranges = [[[aa,.75],[kk,.25]],[[aa,.5],[jj,.5]]];
    // AA/AA is impossible. Remaining pair masses .375/.125/.125 normalize
    // to .6/.2/.2, NOT the original .75/.25 hero marginal.
    for (const [roll,expected] of [[.599,[aa,jj]],[.601,[kk,aa]],[.801,[kk,jj]]] as const) {
      const session = startSpot(fixture,'dealing-test','big-blind',() => roll);
      expect(SPOT_SEATS.map((s) => comboKey(...session.state.holeCards[s]))).toEqual(expected);
      expect(new Set([...session.state.board,...Object.values(session.state.holeCards).flat()]).size).toBe(9);
    }
  });
});
