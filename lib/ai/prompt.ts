// Serialize the current spot into a structured prompt so the model critiques
// the actual situation (ranges, board, equity, any solver output) rather than
// guessing.

export interface SpotContext {
  kind: 'equity' | 'preflop' | 'postflop';
  description: string;
  hero?: string;
  villain?: string;
  board?: string;
  heroRange?: string;
  villainRange?: string;
  equity?: { hero: number; villain: number; tie?: number };
  potBB?: number;
  stackBB?: number;
  extra?: Record<string, string>;
}

export const SYSTEM_PROMPT =
  'You are a world-class Texas Hold\'em poker coach. Analyze the given spot ' +
  'using GTO and exploitative reasoning. Be concise and specific: reference ' +
  'concrete hands, board textures, ranges, and equities. Prefer bullet points. ' +
  'When equity or solver numbers are provided, ground your reasoning in them ' +
  'rather than contradicting them. Avoid generic advice.';

export function buildUserPrompt(spot: SpotContext): string {
  const lines: string[] = [];
  lines.push(`Spot type: ${spot.kind}`);
  if (spot.description) lines.push(spot.description);
  if (spot.hero) lines.push(`Hero position: ${spot.hero}`);
  if (spot.villain) lines.push(`Villain position: ${spot.villain}`);
  if (spot.board) lines.push(`Board: ${spot.board}`);
  if (spot.heroRange) lines.push(`Hero range: ${spot.heroRange}`);
  if (spot.villainRange) lines.push(`Villain range: ${spot.villainRange}`);
  if (spot.equity) {
    lines.push(
      `Equity — Hero ${(spot.equity.hero * 100).toFixed(1)}%, Villain ${(
        spot.equity.villain * 100
      ).toFixed(1)}%` + (spot.equity.tie ? ` (tie ${(spot.equity.tie * 100).toFixed(1)}%)` : '')
    );
  }
  if (spot.potBB != null) lines.push(`Pot: ${spot.potBB}bb`);
  if (spot.stackBB != null) lines.push(`Effective stack: ${spot.stackBB}bb`);
  if (spot.extra) {
    for (const [k, v] of Object.entries(spot.extra)) lines.push(`${k}: ${v}`);
  }
  lines.push('');
  lines.push(
    'Give: (1) a one-line summary of the spot, (2) the recommended strategy ' +
      'and why, (3) key hands/combos and any mixed frequencies, (4) common ' +
      'mistakes to avoid.'
  );
  return lines.join('\n');
}
