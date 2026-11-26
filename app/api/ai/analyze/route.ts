import type { NextRequest } from "next/server";
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

function isSpotContext(value: unknown): value is SpotContext {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const spot = value as Record<string, unknown>;
  if (
    typeof spot.kind !== "string" ||
    !["equity", "preflop", "postflop"].includes(spot.kind) ||
    typeof spot.description !== "string"
  ) {
    return false;
  }
  for (const field of [
    "description", "hero", "villain", "board", "heroRange", "villainRange",
  ]) {
    const entry = spot[field];
    if (entry !== undefined && (typeof entry !== "string" || entry.length > 8000)) {
      return false;
    }
  }
  for (const field of ["potBB", "stackBB"]) {
    const entry = spot[field];
    if (entry !== undefined && (typeof entry !== "number" || !Number.isFinite(entry))) {
      return false;
    }
  }
  if (spot.equity !== undefined) {
    if (!spot.equity || typeof spot.equity !== "object") return false;
    const equity = spot.equity as Record<string, unknown>;
    for (const field of ["hero", "villain", "tie"]) {
      const entry = equity[field];
      if (field === "tie" && entry === undefined) continue;
      if (typeof entry !== "number" || !Number.isFinite(entry) || entry < 0 || entry > 1) {
        return false;
      }
    }
  }
  if (spot.extra !== undefined) {
    if (!spot.extra || typeof spot.extra !== "object" || Array.isArray(spot.extra)) {
      return false;
    }
    const entries = Object.entries(spot.extra);
    if (
      entries.length > 50 ||
      entries.some(([key, entry]) =>
        key.length > 200 || typeof entry !== "string" || entry.length > 8000)
    ) {
      return false;
    }
  }
  return true;
}

export async function POST(req: NextRequest) {
  let body: Body;
  try {
    const value: unknown = await req.json();
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      return json({ error: "Invalid request" }, 400);
    }
    body = value as Body;
  } catch {
    return json({ error: "Invalid JSON" }, 400);
  }
  const { provider, apiKey, model, spot } = body;
  if (
    typeof apiKey !== "string" || !apiKey.trim() ||
    apiKey.length > 4096 || /[\r\n]/.test(apiKey)
  ) {
    return json({ error: "Invalid API key" }, 400);
  }
  if (typeof model !== "string" || !model.trim() || model.length > 200) {
    return json({ error: "Invalid model" }, 400);
  }
  if (!isSpotContext(spot)) return json({ error: "Invalid spot" }, 400);
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
      await upstream.body?.cancel().catch(() => undefined);
      // Provider bodies can echo request credentials. Keep their status, but
      // never reflect the body or transport exception into our response.
      return json(
        {
          error: `Provider request failed (${upstream.status}). Check your API key and model.`,
        },
        upstream.ok ? 502 : upstream.status,
      );
    }

    // Normalize the provider's SSE stream into a plain text token stream.
    const stream = normalizeStream(upstream.body, provider);
    return new Response(stream, {
      headers: {
        "Content-Type": "text/plain; charset=utf-8",
        "Cache-Control": "private, no-store",
      },
    });
  } catch {
    return json({ error: "AI provider request failed. Please try again." }, 502);
  }
}

function json(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json",
      "Cache-Control": "private, no-store",
    },
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
