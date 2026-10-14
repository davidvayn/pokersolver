import type {
  SolverActionTree,
  SolverPlayer,
  SolverTreeNode,
} from '@/lib/solver/client';

export interface SolverLineStep {
  player: SolverPlayer;
  label: string;
  nodeId: number;
  childId: number;
  actionIndex: number;
}

export interface SolverLineSelection {
  node: SolverTreeNode;
  steps: SolverLineStep[];
}

export interface SolverNavigation {
  tree: SolverActionTree | null;
  path: number[];
}

/** Navigate only exported child edges; never synthesize an unmodeled action. */
export function selectSolverLine(
  tree: SolverActionTree,
  path: readonly number[]
): SolverLineSelection | null {
  let node = tree.nodes[tree.root];
  if (!node || node.id !== tree.root) return null;
  const steps: SolverLineStep[] = [];
  for (const actionIndex of path) {
    if (node.kind !== 'decision' || !node.player || !node.strategy ||
        !Number.isInteger(actionIndex) || actionIndex < 0) return null;
    const child = tree.nodes[node.children[actionIndex]];
    const label = node.strategy.actions[actionIndex];
    if (!child || child.id !== node.children[actionIndex] || !label) return null;
    steps.push({ player: node.player, label, nodeId: node.id,
      childId: child.id, actionIndex });
    node = child;
  }
  return { node, steps };
}

/** Keep the saved line through loading and replay it by identity in a new solve. */
export function reconcileSolverNavigation(
  previous: SolverNavigation,
  tree: SolverActionTree | null
): SolverNavigation {
  if (!tree || tree === previous.tree) return previous;
  if (!previous.tree || !previous.path.length) return { tree, path: [] };
  const saved = selectSolverLine(previous.tree, previous.path);
  if (!saved) return previous;
  let node = tree.nodes[tree.root];
  if (!node || node.id !== tree.root) return previous;
  const path: number[] = [];
  for (const step of saved.steps) {
    if (node.kind !== 'decision' || node.player !== step.player || !node.strategy) return previous;
    const matches = node.strategy.actions.flatMap((label, index) => label === step.label ? [index] : []);
    // Ambiguous or absent actions cannot be silently replaced by another size.
    if (matches.length !== 1) return previous;
    const index = matches[0];
    const childId = node.children[index];
    const child = tree.nodes[childId];
    if (!child || child.id !== childId) return previous;
    path.push(index);
    node = child;
  }
  return { tree, path };
}

export function solverAmount(value: number): string {
  return Number(value.toFixed(2)).toString();
}

/** Terminal leaves retain the last actual decision for range review. */
export function strategyNodeForLine(
  tree: SolverActionTree,
  line: SolverLineSelection
): SolverTreeNode | null {
  const id = line.node.kind === 'decision' ? line.node.id : line.steps.at(-1)?.nodeId;
  return id === undefined ? null : tree.nodes[id] ?? null;
}

export function solverActionAmount(
  parent: SolverTreeNode,
  child: SolverTreeNode
): string | null {
  if (!parent.player) return null;
  const player = parent.player === 'oop' ? 0 : 1;
  const added = child.committed[player] - parent.committed[player];
  if (added <= 1e-9) return null;
  return child.committed[player] > parent.committed[1 - player] + 1e-9
    && parent.to_call > 1e-9
    ? `to ${solverAmount(child.committed[player])}bb`
    : `${solverAmount(added)}bb`;
}
