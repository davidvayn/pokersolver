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

export interface AiConversationMessage {
  role: 'user' | 'assistant';
  content: string;
}

const TRANSIENT_SPOT_FIELDS = new Set(['Exploitability', 'OOP EV', 'IP EV']);

/**
 * Stable identity for the inputs a user considers "the spot". Solver
 * diagnostics arrive asynchronously and must not create a new chat thread.
 */
export function buildSpotThreadKey(spot: SpotContext | null): string {
  if (!spot) return 'empty-spot';
  const extra = Object.fromEntries(
    Object.entries(spot.extra ?? {})
      .filter(([key]) => !TRANSIENT_SPOT_FIELDS.has(key))
      .sort(([left], [right]) => left.localeCompare(right))
  );
  return JSON.stringify({
    kind: spot.kind,
    board: spot.board ?? '',
    hero: spot.hero ?? '',
    villain: spot.villain ?? '',
    heroRange: spot.heroRange ?? '',
    villainRange: spot.villainRange ?? '',
    potBB: spot.potBB ?? null,
    stackBB: spot.stackBB ?? null,
    extra,
  });
}

export function describeSpot(spot: SpotContext | null): string {
  if (!spot) return 'Incomplete spot';
  const board = spot.board?.match(/.{1,2}/g)?.join(' ') || 'No board';
  const pot = spot.potBB == null ? null : `${spot.potBB}bb pot`;
  return [board, pot].filter(Boolean).join(' · ');
}

/** Keep follow-up context bounded and provider-safe at the API boundary. */
export function normalizeConversation(
  value: unknown,
  maxMessages = 12
): AiConversationMessage[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter(
      (message): message is AiConversationMessage =>
        Boolean(message) &&
        typeof message === 'object' &&
        ((message as AiConversationMessage).role === 'user' ||
          (message as AiConversationMessage).role === 'assistant') &&
        typeof (message as AiConversationMessage).content === 'string'
    )
    .map((message) => ({
      role: message.role,
      content: message.content.trim().slice(0, 8000),
    }))
    .filter((message) => message.content.length > 0)
    .slice(-maxMessages);
}

export const SYSTEM_PROMPT =
  'You are a world-class Texas Hold\'em poker coach. Analyze the given spot ' +
  'using GTO and exploitative reasoning. Be concise and specific: reference ' +
  'concrete hands, board textures, ranges, and equities. Prefer bullet points. ' +
  'When equity or solver numbers are provided, ground your reasoning in them ' +
  'rather than contradicting them. Format responses as Markdown with short ' +
  'headings and bullet lists. Use plain card notation such as Qh, 7s, and 2c; ' +
  'do not use LaTeX. Avoid generic advice.';

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
