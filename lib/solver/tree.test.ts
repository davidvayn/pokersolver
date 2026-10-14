import { describe, expect, it } from 'vitest';
import { reconcileSolverNavigation, selectSolverLine, solverActionAmount, solverAmount, strategyNodeForLine } from '@/lib/solver/tree';
import type { SolverActionTree, SolverTreeNode } from '@/lib/solver/client';

function fixture(): SolverActionTree {
  const base: SolverTreeNode = {
    id: 0, kind: 'decision', player: 'oop', pot: 6,
    stacks: [100, 100], committed: [0, 0], to_call: 0, reach_probability: 1,
    children: [], strategy: null, reason: null, winner: null,
  };
  const decision = (id: number, player: 'oop' | 'ip', pot: number,
    committed: [number, number], actions: string[], children: number[]): SolverTreeNode => ({
    ...base, id, player, pot, committed,
    stacks: [100 - committed[0], 100 - committed[1]],
    to_call: Math.max(0, committed[player === 'oop' ? 1 : 0] - committed[player === 'oop' ? 0 : 1]),
    strategy: { title: player, actions, rows: [] }, children,
  });
  const terminal = (id: number, pot: number, committed: [number, number],
    winner: 'oop' | 'ip' | null = null): SolverTreeNode => ({
    ...base, id, pot, committed, stacks: [100 - committed[0], 100 - committed[1]],
    kind: 'terminal', player: null, reason: winner ? 'fold' : 'showdown', winner,
  });
  return { root: 0, max_raises: 3, nodes: [
    decision(0, 'oop', 6, [0, 0], ['Check', 'Bet 50%'], [1, 3]),
    decision(1, 'ip', 6, [0, 0], ['Check'], [2]),
    terminal(2, 6, [0, 0]),
    decision(3, 'ip', 9, [3, 0], ['Fold', 'Call', 'Raise 100%'], [4, 5, 6]),
    terminal(4, 9, [3, 0], 'oop'),
    terminal(5, 12, [3, 3]),
    decision(6, 'oop', 24, [3, 15], ['Fold', 'Call'], [7, 8]),
    terminal(7, 24, [3, 15], 'ip'),
    terminal(8, 36, [15, 15]),
  ] };
}

describe('solved betting-tree navigation', () => {
  it('keeps the complete line while a board edit clears the result and after its new solve arrives', () => {
    const tree = fixture();
    const navigation = { tree, path: [1, 2, 1] };
    const pending = reconcileSolverNavigation(navigation, null);
    expect(pending).toBe(navigation);
    const refreshed = reconcileSolverNavigation(pending, fixture());
    expect(refreshed.path).toEqual(navigation.path);
    const line = selectSolverLine(refreshed.tree!, refreshed.path)!;
    expect(line.steps.map((step) => step.label)).toEqual(['Bet 50%', 'Raise 100%', 'Call']);
    expect(line.node.kind).toBe('terminal');
    expect(line.node.pot).toBe(36);
  });

  it('replays saved action names and actors, not stale action indices, in a refreshed tree', () => {
    const tree = fixture();
    const refreshedTree = fixture();
    refreshedTree.nodes[0].strategy!.actions.reverse();
    refreshedTree.nodes[0].children.reverse();
    const snapshot = JSON.stringify(refreshedTree);
    const navigation = reconcileSolverNavigation({ tree, path: [1, 2] }, refreshedTree);
    expect(navigation.path).toEqual([0, 2]);
    expect(selectSolverLine(navigation.tree!, navigation.path)?.node.id).toBe(6);
    expect(JSON.stringify(refreshedTree)).toBe(snapshot);
  });

  it('retains every saved action if a settings change makes the line unmodeled instead of silently resetting', () => {
    const tree = fixture();
    const navigation = { tree, path: [1, 2] };
    const differentSizes = fixture();
    differentSizes.nodes[0].strategy!.actions[1] = 'Bet 75%';
    expect(reconcileSolverNavigation(navigation, differentSizes)).toBe(navigation);
    const differentActor = fixture();
    differentActor.nodes[3].player = 'oop';
    expect(reconcileSolverNavigation(navigation, differentActor)).toBe(navigation);
    const malformedTree = fixture();
    malformedTree.nodes[3].children[2] = 99;
    expect(reconcileSolverNavigation(navigation, malformedTree)).toBe(navigation);
    const ambiguousTree = fixture();
    ambiguousTree.nodes[0].strategy!.actions[0] = 'Bet 50%';
    expect(reconcileSolverNavigation(navigation, ambiguousTree)).toBe(navigation);
  });

  it('starts at the refreshed root only after an explicit empty-path reset or the initial solve', () => {
    const tree = fixture();
    const initial = reconcileSolverNavigation({ tree: null, path: [] }, tree);
    expect(initial).toEqual({ tree, path: [] });
    const reset = reconcileSolverNavigation({ tree: fixture(), path: [] }, tree);
    expect(reset).toEqual(initial);
    expect(reconcileSolverNavigation(reset, null)).toBe(reset);
    expect(reconcileSolverNavigation(reset, tree)).toBe(reset);
  });

  it('retains the last real decision strategy at terminal leaves', () => {
    const tree = fixture();
    const called = selectSolverLine(tree, [1, 2, 1])!;
    const folded = selectSolverLine(tree, [1, 2, 0])!;
    expect(strategyNodeForLine(tree, called)).toBe(tree.nodes[6]);
    expect(strategyNodeForLine(tree, folded)).toBe(tree.nodes[6]);
    expect(strategyNodeForLine(tree, selectSolverLine(tree, [])!)).toBe(tree.nodes[0]);
  });
  it('follows a bet, raise and call, switching actors and preserving the original result', () => {
    const tree = fixture();
    const before = JSON.stringify(tree);
    const raised = selectSolverLine(tree, [1, 2]);
    expect(raised?.node.player).toBe('oop');
    expect(raised?.node.pot).toBe(24);
    expect(raised?.node.stacks).toEqual([97, 85]);
    expect(raised?.steps.map((step) => [step.player, step.label])).toEqual([
      ['oop', 'Bet 50%'], ['ip', 'Raise 100%'],
    ]);
    expect(selectSolverLine(tree, [1, 2, 1])?.node.reason).toBe('showdown');
    expect(selectSolverLine(tree, [1, 2, 0])?.node.winner).toBe('ip');
    expect(selectSolverLine(tree, [0, 0])?.node.reason).toBe('showdown');
    expect(JSON.stringify(tree)).toBe(before);
  });

  it('navigates zero-frequency branches without synthesizing missing actions', () => {
    const tree = fixture();
    tree.nodes[3].reach_probability = 0;
    expect(selectSolverLine(tree, [1])?.node.strategy?.actions).toEqual(['Fold', 'Call', 'Raise 100%']);
    expect(selectSolverLine(tree, [1, 2])?.node.id).toBe(6);
    for (const path of [[-1], [2], [NaN], [0.5], [1, 2, 1, 0]]) {
      expect(selectSolverLine(tree, path)).toBeNull();
    }
    tree.nodes[3].children[2] = 99;
    expect(selectSolverLine(tree, [1, 2])).toBeNull();
  });

  it('rewinds a line by selecting a shorter path and rejects stale node identities', () => {
    const tree = fixture();
    const path = [1, 2];
    expect(selectSolverLine(tree, path.slice(0, 1))?.node.id).toBe(3);
    expect(selectSolverLine(tree, [])?.node.id).toBe(0);
    tree.nodes[0].id = 22;
    expect(selectSolverLine(tree, [])).toBeNull();
  });

  it('shows actual chip amounts rather than treating raise percentages as bet amounts', () => {
    const { nodes } = fixture();
    expect(solverActionAmount(nodes[0], nodes[3])).toBe('3bb');
    expect(solverActionAmount(nodes[3], nodes[5])).toBe('3bb');
    expect(solverActionAmount(nodes[3], nodes[6])).toBe('to 15bb');
    expect(solverActionAmount(nodes[3], nodes[4])).toBeNull();
    expect(solverActionAmount(nodes[0], nodes[1])).toBeNull();
    expect(solverAmount(1.98000000001)).toBe('1.98');
  });
});
