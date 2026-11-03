import { beginSpotAction, prepareSpotSession, spotActions,
  type SolvedSpot, type SpotSession } from '@/lib/solved-spots';

/** Publish the pending engine state before fetching its strategy. On failure the
 * last published session is the retry cursor, not the pre-action session. */
export async function prepareSpotLine(
  spot: SolvedSpot,
  pending: SpotSession,
  stop: 'feedback' | 'decision',
  publish: (session: SpotSession) => void,
  random = Math.random,
): Promise<SpotSession> {
  let current = pending;
  for (let step = 0; step < 16; step++) {
    publish(current);
    current = await prepareSpotSession(spot, current);
    publish(current);
    if (stop === 'feedback' || current.state.terminal || current.state.toAct === current.state.hero) return current;
    const actions = spotActions(spot, current);
    const unit = random();
    if (!Number.isFinite(unit) || unit < 0 || unit >= 1) throw new Error('Invalid RNG value');
    let roll = unit * actions.reduce((sum, action) => sum + action.probability, 0);
    const action = actions.find((action) => (roll -= action.probability) < 0) ?? actions.at(-1)!;
    current = beginSpotAction(spot, current, action.id);
  }
  throw new Error('Saved opponent line exceeds the bounded game');
}
