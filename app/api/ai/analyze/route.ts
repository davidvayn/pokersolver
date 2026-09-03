import { NextRequest } from "next/server";
import {
  SYSTEM_PROMPT,
  buildUserPrompt,
  normalizeConversation,
  type AiConversationMessage,
  type SpotContext,
} from "@/lib/ai/prompt";
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
    const upstream =
      provider === "anthropic"
        ? await callAnthropic(apiKey, model, userPrompt, conversation)
        : provider === "gemini"
          ? await callGemini(apiKey, model, userPrompt, conversation)
          : await callOpenAI(apiKey, model, userPrompt, conversation);

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
        generationConfig: { maxOutputTokens: 2048 },
      }),
    },
  );
}

/** Parse provider SSE and emit only the text deltas as a UTF-8 stream. */
function normalizeStream(
  body: ReadableStream<Uint8Array>,
  provider: ProviderId,
): ReadableStream<Uint8Array> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = "";
  let pendingCarriageReturn = false;

  // SSE permits CRLF, lone LF, and lone CR line endings. Canonicalize them
  // without mistaking a CRLF pair split across network chunks for a blank line.
  const normalizeLineEndings = (chunk: string, flush = false): string => {
    let text = chunk;
    if (pendingCarriageReturn) {
      if (text.startsWith("\n")) text = text.slice(1);
      text = `\n${text}`;
      pendingCarriageReturn = false;
    }
    if (!flush && text.endsWith("\r")) {
      text = text.slice(0, -1);
      pendingCarriageReturn = true;
    }
    return text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  };

  const processEvents = (
    block: string,
    controller: ReadableStreamDefaultController<Uint8Array>,
  ): boolean => {
    for (const evt of block.split("\n\n")) {
      const data = evt
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n")
        .trim();
      if (!data) continue;
      if (data === "[DONE]") {
        controller.close();
        void reader.cancel();
        return true;
      }
      try {
        const obj = JSON.parse(data);
        // Surface provider-side errors instead of silently dropping them.
        const errMsg =
          obj?.type === "error"
            ? obj.error?.message || "provider stream error"
            : obj?.error
              ? obj.error.message || String(obj.error)
              : null;
        if (errMsg) {
          controller.enqueue(
            encoder.encode(`\n\n⚠️ AI provider error: ${errMsg}`),
          );
          controller.close();
          reader.cancel();
          return true; // stop
        }
        const text =
          provider === "anthropic"
            ? obj.type === "content_block_delta"
              ? (obj.delta?.text ?? "")
              : ""
            : provider === "gemini"
              ? (obj.candidates?.[0]?.content?.parts ?? [])
                  .map((part: { text?: string }) => part.text ?? "")
                  .join("")
              : (obj.choices?.[0]?.delta?.content ?? "");
        if (text) controller.enqueue(encoder.encode(text));
        const streamFinished =
          (provider === "gemini" &&
            Boolean(obj.candidates?.[0]?.finishReason)) ||
          (provider === "anthropic" && obj.type === "message_stop");
        if (streamFinished) {
          controller.close();
          void reader.cancel();
          return true;
        }
      } catch {
        /* skip non-JSON keepalive events */
      }
    }
    return false;
  };

  return new ReadableStream({
    async pull(controller) {
      const { done, value } = await reader.read();
      if (done) {
        // Flush any residual buffered event before closing.
        buffer += normalizeLineEndings(decoder.decode(), true);
        if (buffer.trim() && processEvents(buffer, controller)) return;
        controller.close();
        return;
      }
      buffer += normalizeLineEndings(decoder.decode(value, { stream: true }));
      const events = buffer.split("\n\n");
      buffer = events.pop() ?? "";
      if (processEvents(events.join("\n\n"), controller)) return;
    },
    cancel() {
      reader.cancel();
    },
  });
}
