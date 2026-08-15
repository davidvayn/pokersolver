import type {
  ConfidenceLevel,
  EvGrade,
  PolicyAction,
  PolicyNode,
} from '@/lib/practice-types';

export interface GradeResult {
  chosenActionEvBb: number | null;
  bestActionEvBb: number | null;
  evLossBb: number | null;
  grade: EvGrade;
  confidence: ConfidenceLevel;
  lowConfidence: boolean;
}

export function gradeEvLoss(loss: number | null): EvGrade {
  if (loss === null || !Number.isFinite(loss)) return 'ungraded';
  const normalized = Math.max(0, loss);
  if (normalized <= 0.01) return 'optimal';
  if (normalized <= 0.05) return 'good';
  if (normalized <= 0.25) return 'mistake';
  return 'blunder';
}

export function gradePolicyChoice(
  node: PolicyNode,
  chosenActionId: string
): GradeResult {
  const chosen = node.actions.find((action) => action.id === chosenActionId);
  if (!chosen) throw new Error('Chosen action is absent from the policy node');
  const finiteValues = node.actions.filter(
    (action): action is PolicyAction & { evBb: number } =>
      action.evBb !== null && Number.isFinite(action.evBb)
  );
  const best =
    node.bestActionEvBb ??
    (finiteValues.length > 0
      ? Math.max(...finiteValues.map((action) => action.evBb))
      : null);
  const loss =
    chosen.evBb === null || best === null
      ? null
      : Math.max(0, best - chosen.evBb);
  const confidence = chosen.confidence;
  return {
    chosenActionEvBb: chosen.evBb,
    bestActionEvBb: best,
    evLossBb: loss,
    grade: gradeEvLoss(loss),
    confidence,
    lowConfidence: confidence !== 'high',
  };
}

export function validatePolicyNode(node: PolicyNode): string[] {
  const errors: string[] = [];
  if (!/^[a-f0-9]{64}$/.test(node.stateHash)) {
    errors.push('State hash must be a 64-character SHA-256 hex digest');
  }
  if (node.actions.length === 0) errors.push('Policy node has no actions');
  const ids = new Set<string>();
  let sum = 0;
  for (const action of node.actions) {
    if (ids.has(action.id)) errors.push(`Duplicate action ${action.id}`);
    ids.add(action.id);
    if (!Number.isFinite(action.probability) || action.probability < 0) {
      errors.push(`Invalid probability for ${action.id}`);
    } else {
      sum += action.probability;
    }
    if (!['high', 'low', 'unavailable'].includes(action.confidence)) {
      errors.push(`Invalid confidence for ${action.id}`);
    }
    const invalidEv =
      action.evBb === null
        ? action.standardErrorBb !== null
        : !Number.isFinite(action.evBb) ||
          (action.standardErrorBb !== null &&
            (!Number.isFinite(action.standardErrorBb) ||
              action.standardErrorBb < 0)) ||
          (action.standardErrorBb === null && action.confidence === 'high');
    if (invalidEv) {
      errors.push(`Invalid action EV data for ${action.id}`);
    }
  }
  if (Math.abs(sum - 1) > 1e-6) {
    errors.push(`Action probabilities sum to ${sum}`);
  }
  if (node.bestActionId !== null && !ids.has(node.bestActionId)) {
    errors.push('Best action is absent from the action list');
  }
  return errors;
}

export function samplePolicyAction<T extends { probability: number }>(
  actions: T[],
  random: () => number = Math.random
): T {
  if (actions.length === 0) throw new Error('Cannot sample an empty policy');
  const roll = random();
  let cumulative = 0;
  for (const action of actions) {
    cumulative += action.probability;
    if (roll < cumulative) return action;
  }
  return actions[actions.length - 1];
}
