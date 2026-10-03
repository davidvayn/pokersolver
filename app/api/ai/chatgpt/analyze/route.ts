import { NextRequest } from 'next/server';
import { accessToken, accountModels, assertLocalRequest, getSession } from '@/lib/server/chatgpt-auth';
import { buildUserPrompt, normalizeConversation, SYSTEM_PROMPT, type SpotContext } from '@/lib/ai/prompt';
import { normalizeStream } from '@/lib/ai/stream';

export const runtime = 'nodejs';

export async function POST(req: NextRequest) {
  try {
    assertLocalRequest(req, true);
    const session = getSession(req);
    if (!session) return Response.json({ error: 'Sign in with ChatGPT in Settings first.' }, { status: 401 });
    const body = await req.json() as { model: string; spot: SpotContext; messages?: unknown };
    if (!body.spot) return Response.json({ error: 'Missing spot.' }, { status: 400 });
    const models = await accountModels(session);
    if (!models.some((model) => model.id === body.model)) {
      return Response.json({ error: 'Choose a model available to your ChatGPT account in Settings.' }, { status: 400 });
    }
    const upstream = await fetch('https://api.openai.com/v1/responses', {
      method: 'POST',
      headers: { Authorization: `Bearer ${await accessToken(session)}`, 'Content-Type': 'application/json' },
      signal: req.signal,
      body: JSON.stringify({
        model: body.model, store: false, stream: true,
        instructions: SYSTEM_PROMPT,
        input: [{ role: 'user', content: buildUserPrompt(body.spot) }, ...normalizeConversation(body.messages)],
      }),
    });
    if (!upstream.ok || !upstream.body) {
      return Response.json({ error: `ChatGPT rejected the request (${upstream.status}). Check your plan access and usage limits in ChatGPT Settings.` }, { status: upstream.status || 502 });
    }
    return new Response(normalizeStream(upstream.body, 'responses'), {
      headers: { 'Content-Type': 'text/plain; charset=utf-8', 'Cache-Control': 'no-store' },
    });
  } catch (caught) {
    return Response.json({ error: (caught as Error).message }, { status: 502 });
  }
}
