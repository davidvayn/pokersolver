import { describe, expect, it } from 'vitest';
import { parseBoard } from '@/lib/cards';
import { createHand, applyAction, seededRandom, totalPotBb } from '@/lib/practice-engine';
import { neuralLegalActions } from '@/lib/neural-policy';
import { ACTIVE_FULL_HAND_MANIFESTS } from '@/lib/practice-models';
import { replayStudyLine, studyDecisionKey, type StudyChoice, type StudyConfig } from '@/lib/solver/study-engine';
import type { ActionKind } from '@/lib/practice-types';

const manifest = ACTIVE_FULL_HAND_MANIFESTS[0];
const config: StudyConfig = { modelVersion: manifest.version, depthBb: 20,
  actionAbstraction: manifest.runtime!.kind === 'rust-continual-resolver-v1' ? manifest.runtime!.actionAbstraction : (() => { throw new Error('missing runtime'); })() };
const board = parseBoard('Qh 7s 2c 5h 3d');
function choose(choices: StudyChoice[], kind: ActionKind, target?: number): StudyChoice[] {
  const line = replayStudyLine(config, board, choices);
  const decision = line.decisions.at(-1)!;
  const action = decision.actions.find((a) => a.kind === kind && (target === undefined || a.amountToBb === target));
  if (!action) throw new Error(`missing ${kind} at ${line.state.street}`);
  return [...choices, { street: decision.state.street, actor: decision.state.toAct!, action }];
}
function opened(): StudyChoice[] { return choose(choose([], 'raise', 3), 'call'); }

describe('shared multi-street study engine', () => {
  it('uses the same modeled actions and preflop chip accounting as Practice', () => {
    let practice = createHand({ modelVersion: config.modelVersion, depthBb: 20,
      button: 'button-small-blind', hero: 'button-small-blind', random: seededRandom(9) });
    let choices: StudyChoice[] = [];
    for (const [kind, target] of [['raise', 3], ['call', undefined]] as const) {
      const practiceActions = neuralLegalActions(practice, config.actionAbstraction);
      const study = replayStudyLine(config, board, choices);
      expect(study.decisions.at(-1)!.actions).toEqual(practiceActions);
      const action = practiceActions.find((a) => a.kind === kind && (target === undefined || a.amountToBb === target))!;
      choices = choose(choices, kind, target);
      practice = applyAction(practice, action);
    }
    const study = replayStudyLine(config, board, choices).state;
    expect(study.street).toBe('flop');
    expect(study.toAct).toBe('big-blind');
    expect(study.stacksBb).toEqual(practice.stacksBb);
    expect(study.actionHistory).toEqual(practice.actionHistory.map((a) => ({ ...a, id: a.id.replace(practice.id, study.id) })));
    expect(totalPotBb(study)).toBe(6);
  });

  it('continues flop/turn/river betting, keeps prior actions and never invents a private showdown winner', () => {
    let choices = opened();
    choices = choose(choose(choices, 'bet', 2), 'call');
    expect(replayStudyLine(config, board, choices).state.street).toBe('turn');
    expect(totalPotBb(replayStudyLine(config, board, choices).state)).toBe(10);
    choices = choose(choose(choices, 'check'), 'check');
    expect(replayStudyLine(config, board, choices).state.street).toBe('river');
    choices = choose(choose(choices, 'bet', 5), 'call');
    const line = replayStudyLine(config, board, choices);
    expect(line.state.terminal).toBe(true);
    expect(line.state.result).toMatchObject({ reason: 'review-complete', winner: null, potBb: 20 });
    expect(line.state.actionHistory).toHaveLength(8);
    expect(totalPotBb(line.state) + Object.values(line.state.stacksBb).reduce((a, b) => a + b, 0)).toBe(40);
    const changed = replayStudyLine(config, parseBoard('Qh 7s 2c 6h 4d'), choices);
    expect(changed.state.actionHistory).toEqual(line.state.actionHistory);
    expect(changed.state.stacksBb).toEqual(line.state.stacksBb);
  });

  it('pauses for missing public cards and replays the saved continuation when those cards return', () => {
    let choices = choose(choose(opened(), 'check'), 'check');
    choices = choose(choose(choices, 'check'), 'check');
    const snapshot = JSON.stringify(choices);
    const paused = replayStudyLine(config, board.slice(0, 3), choices);
    expect(paused.waitingForCards).toBe(4);
    expect(paused.state.toAct).toBeNull();
    expect(paused.unappliedChoices).toBe(2);
    expect(JSON.stringify(choices)).toBe(snapshot);
    const resumed = replayStudyLine(config, board, choices);
    expect(resumed.state.street).toBe('river');
    expect(resumed.unappliedChoices).toBe(0);
    expect(resumed.state.actionHistory).toHaveLength(6);
  });

  it('handles folds and all-in runouts without offering another betting round', () => {
    const folded = replayStudyLine(config, board, choose([], 'fold'));
    expect(folded.state.result).toMatchObject({ reason: 'fold', winner: 'big-blind' });
    const choices = choose(choose([], 'all-in'), 'call');
    const paused = replayStudyLine(config, board.slice(0, 3), choices);
    expect(paused.waitingForCards).toBe(4);
    expect(paused.state.toAct).toBeNull();
    expect(paused.state.stacksBb).toEqual({ 'button-small-blind': 0, 'big-blind': 0 });
    expect(replayStudyLine(config, board, choices).state.result?.winner).toBeNull();
  });

  it('supports deeper rule states without implying trained coverage and rejects duplicate cards', () => {
    for (const depthBb of [50, 100]) {
      const line = replayStudyLine({ ...config, depthBb }, [], []);
      expect(totalPotBb(line.state)).toBe(1.5);
      expect(line.state.stacksBb['button-small-blind']).toBe(depthBb - .5);
    }
    expect(() => replayStudyLine(config, [0, 0, 1], [])).toThrow('unique');
    const a = replayStudyLine(config, board, []).state;
    const b = { ...a, holeCards: { ...a.holeCards, 'button-small-blind': [48, 49] as [number, number] } };
    expect(studyDecisionKey(a)).toBe(studyDecisionKey(b));
  });
});
