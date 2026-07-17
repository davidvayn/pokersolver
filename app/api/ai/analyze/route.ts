import { NextRequest } from 'next/server';
import { SYSTEM_PROMPT, buildUserPrompt, SpotContext } from '@/lib/ai/prompt';
import { ProviderId } from '@/lib/ai/providers';

export const runtime = 'edge';

interface Body {
  provider: ProviderId;
  apiKey: string;
  model: string;
  spot: SpotContext;
}

export async function POST(req: NextRequest) {
  let body: Body;
  try {
    body = (await req.json()) as Body;
  } catch {
    return json({ error: 'Invalid JSON' }, 400);
  }
  const { provider, apiKey, model, spot } = body;
  if (!apiKey) return json({ error: 'Missing API key' }, 400);
  if (!spot) return json({ error: 'Missing spot' }, 400);

  const userPrompt = buildUserPrompt(spot);

  try {
    const upstream =
      provider === 'anthropic'
        ? await callAnthropic(apiKey, model, userPrompt)
        : await callOpenAI(apiKey, model, userPrompt);

    if (!upstream.ok || !upstream.body) {
      const text = await upstream.text().catch(() => '');
      return json(
        { error: `Provider error (${upstream.status}): ${text.slice(0, 500)}` },
        upstream.status || 502
      );
    }

    // Normalize the provider's SSE stream into a plain text token stream.
    const stream = normalizeStream(upstream.body, provider);
    return new Response(stream, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Cache-Control': 'no-cache',
      },
    });
  } catch (e) {
    return json({ error: `Request failed: ${(e as Error).message}` }, 502);
  }
}

function json(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function callAnthropic(apiKey: string, model: string, prompt: string) {
  return fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'content-type': 'application/json',
    },
    body: JSON.stringify({
      model,
      max_tokens: 2048,
      system: SYSTEM_PROMPT,
      stream: true,
      messages: [{ role: 'user', content: prompt }],
    }),
  });
}

function callOpenAI(apiKey: string, model: string, prompt: string) {
  return fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'content-type': 'application/json',
    },
    body: JSON.stringify({
      model,
      stream: true,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: prompt },
      ],
    }),
  });
}

/** Parse provider SSE and emit only the text deltas as a UTF-8 stream. */
function normalizeStream(
  body: ReadableStream<Uint8Array>,
  provider: ProviderId
): ReadableStream<Uint8Array> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = '';

  const processEvents = (
    block: string,
    controller: ReadableStreamDefaultController<Uint8Array>
  ): boolean => {
    for (const evt of block.split('\n\n')) {
      for (const line of evt.split('\n')) {
        if (!line.startsWith('data:')) continue;
        const data = line.slice(5).trim();
        if (!data || data === '[DONE]') continue;
        try {
          const obj = JSON.parse(data);
          // Surface provider-side errors instead of silently dropping them.
          const errMsg =
            obj?.type === 'error'
              ? obj.error?.message || 'provider stream error'
              : obj?.error
                ? obj.error.message || String(obj.error)
                : null;
          if (errMsg) {
            controller.enqueue(encoder.encode(`\n\n⚠️ AI provider error: ${errMsg}`));
            controller.close();
            reader.cancel();
            return true; // stop
          }
          const text =
            provider === 'anthropic'
              ? obj.type === 'content_block_delta'
                ? obj.delta?.text ?? ''
                : ''
              : obj.choices?.[0]?.delta?.content ?? '';
          if (text) controller.enqueue(encoder.encode(text));
        } catch {
          /* skip non-JSON keepalive lines */
        }
      }
    }
    return false;
  };

  return new ReadableStream({
    async pull(controller) {
      const { done, value } = await reader.read();
      if (done) {
        // Flush any residual buffered event before closing.
        buffer += decoder.decode();
        if (buffer.trim()) processEvents(buffer, controller);
        controller.close();
        return;
      }
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split('\n\n');
      buffer = events.pop() ?? '';
      processEvents(events.join('\n\n'), controller);
    },
    cancel() {
      reader.cancel();
    },
  });
}
