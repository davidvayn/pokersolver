import { NextRequest } from "next/server";
import {
  SYSTEM_PROMPT,
  buildUserPrompt,
  normalizeConversation,
  type AiConversationMessage,
  type SpotContext,
} from "@/lib/ai/prompt";
import { normalizeStream } from "@/lib/ai/stream";
import type { ProviderId } from "@/lib/ai/providers";

export const runtime = "edge";

interface Body {
  provider: ProviderId;
  apiKey: string;
  model: string;
  spot: SpotContext;
  messages?: unknown;
}

export async function POST(req: NextRequest) {
  let body: Body;
  try {
    body = (await req.json()) as Body;
  } catch {
    return json({ error: "Invalid JSON" }, 400);
  }
  const { provider, apiKey, model, spot } = body;
  if (!apiKey) return json({ error: "Missing API key" }, 400);
  if (!spot) return json({ error: "Missing spot" }, 400);
  if (!["anthropic", "openai", "gemini"].includes(provider)) {
    return json({ error: "Unsupported provider" }, 400);
  }

  const userPrompt = buildUserPrompt(spot);
  const conversation = normalizeConversation(body.messages);

  try {
    const requestProvider = () =>
      provider === "anthropic"
        ? callAnthropic(apiKey, model, userPrompt, conversation)
        : provider === "gemini"
          ? callGemini(apiKey, model, userPrompt, conversation)
          : callOpenAI(apiKey, model, userPrompt, conversation);
    let upstream = await requestProvider();
    // Retry rejected requests before any response text reaches the browser.
    // Keep the delay short enough for the client's first-token timeout.
    for (
      let attempt = 0;
      attempt < 2 && [429, 503].includes(upstream.status);
      attempt += 1
    ) {
      await upstream.body?.cancel();
      await new Promise((resolve) => setTimeout(resolve, 250 * (attempt + 1)));
      if (req.signal.aborted) throw new Error("Request canceled");
      upstream = await requestProvider();
    }

    if (!upstream.ok || !upstream.body) {
      const text = await upstream.text().catch(() => "");
      return json(
        { error: `Provider error (${upstream.status}): ${text.slice(0, 500)}` },
        upstream.status || 502,
      );
    }

    // Normalize the provider's SSE stream into a plain text token stream.
    const stream = normalizeStream(upstream.body, provider);
    return new Response(stream, {
      headers: {
        "Content-Type": "text/plain; charset=utf-8",
        "Cache-Control": "no-cache",
      },
    });
  } catch (e) {
    return json({ error: `Request failed: ${(e as Error).message}` }, 502);
  }
}

function json(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function providerMessages(
  prompt: string,
  conversation: AiConversationMessage[],
): AiConversationMessage[] {
  return [{ role: "user", content: prompt }, ...conversation];
}

function callAnthropic(
  apiKey: string,
  model: string,
  prompt: string,
  conversation: AiConversationMessage[],
) {
  return fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model,
      max_tokens: 2048,
      system: SYSTEM_PROMPT,
      stream: true,
      messages: providerMessages(prompt, conversation),
    }),
  });
}

function callOpenAI(
  apiKey: string,
  model: string,
  prompt: string,
  conversation: AiConversationMessage[],
) {
  return fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model,
      stream: true,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        ...providerMessages(prompt, conversation),
      ],
    }),
  });
}

function callGemini(
  apiKey: string,
  model: string,
  prompt: string,
  conversation: AiConversationMessage[],
) {
  const contents = providerMessages(prompt, conversation).map((message) => ({
    role: message.role === "assistant" ? "model" : "user",
    parts: [{ text: message.content }],
  }));
  return fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:streamGenerateContent?alt=sse`,
    {
      method: "POST",
      headers: {
        "x-goog-api-key": apiKey,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        system_instruction: { parts: [{ text: SYSTEM_PROMPT }] },
        contents,
        // Gemini's output limit includes its internal reasoning tokens.
        generationConfig: {
          maxOutputTokens: 8192,
          ...(model.startsWith("gemini-3")
            ? { thinkingConfig: { thinkingLevel: "low" } }
            : model.startsWith("gemini-2.5")
              ? { thinkingConfig: { thinkingBudget: 2048 } }
              : {}),
        },
      }),
    },
  );
}
