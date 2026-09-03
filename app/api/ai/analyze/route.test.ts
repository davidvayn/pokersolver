import { afterEach, describe, expect, it, vi } from "vitest";
import { NextRequest } from "next/server";
import { POST } from "./route";

const SPOT = {
  kind: "postflop" as const,
  description: "Test spot",
  board: "Qh7s2c",
  heroRange: "AA,KK",
  villainRange: "QQ,JJ",
  potBB: 6,
  stackBB: 100,
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("AI analysis route conversation", () => {
  it("sends the spot followed by the existing thread to OpenAI", async () => {
    const upstreamFetch = vi
      .fn()
      .mockResolvedValue(
        new Response(
          'data: {"choices":[{"delta":{"content":"Follow-up"}}]}\n\n' +
            "data: [DONE]\n\n",
          { headers: { "Content-Type": "text/event-stream" } },
        ),
      );
    vi.stubGlobal("fetch", upstreamFetch);

    const response = await POST(
      new NextRequest("http://localhost/api/ai/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: "openai",
          apiKey: "test-key",
          model: "test-model",
          spot: SPOT,
          messages: [
            { role: "assistant", content: "Initial analysis" },
            { role: "user", content: "Why this sizing?" },
          ],
        }),
      }),
    );

    expect(await response.text()).toBe("Follow-up");
    const request = upstreamFetch.mock.calls[0][1] as RequestInit;
    const payload = JSON.parse(String(request.body));
    expect(
      payload.messages.map((message: { role: string }) => message.role),
    ).toEqual(["system", "user", "assistant", "user"]);
    expect(payload.messages[1].content).toContain("Board: Qh7s2c");
    expect(payload.messages[2].content).toBe("Initial analysis");
    expect(payload.messages[3].content).toBe("Why this sizing?");
  });

  it("maps the conversation and streaming response for Gemini", async () => {
    const upstreamFetch = vi
      .fn()
      .mockResolvedValue(
        new Response(
          'data: {"candidates":[{"content":{"parts":[{"text":"Gemini reply"}]}}]}\n\n',
          { headers: { "Content-Type": "text/event-stream" } },
        ),
      );
    vi.stubGlobal("fetch", upstreamFetch);

    const response = await POST(
      new NextRequest("http://localhost/api/ai/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: "gemini",
          apiKey: "gemini-test-key",
          model: "gemini-3.7-flash",
          spot: SPOT,
          messages: [
            { role: "assistant", content: "Initial analysis" },
            { role: "user", content: "Why this sizing?" },
          ],
        }),
      }),
    );

    expect(await response.text()).toBe("Gemini reply");
    expect(upstreamFetch.mock.calls[0][0]).toBe(
      "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:streamGenerateContent?alt=sse",
    );
    const request = upstreamFetch.mock.calls[0][1] as RequestInit;
    expect(new Headers(request.headers).get("x-goog-api-key")).toBe(
      "gemini-test-key",
    );
    const payload = JSON.parse(String(request.body));
    expect(payload.system_instruction.parts[0].text).toContain("poker coach");
    expect(payload.contents).toEqual([
      expect.objectContaining({ role: "user" }),
      { role: "model", parts: [{ text: "Initial analysis" }] },
      { role: "user", parts: [{ text: "Why this sizing?" }] },
    ]);
  });

  it("forwards a CRLF-framed Gemini event before the upstream stream closes", async () => {
    const encoder = new TextEncoder();
    const upstream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(
          encoder.encode(
            'data: {"candidates":[{"content":{"parts":[{"text":"First token"}]}}]}\r\n\r\n',
          ),
        );
      },
    });
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(upstream, {
          headers: { "Content-Type": "text/event-stream" },
        }),
      ),
    );

    const response = await POST(
      new NextRequest("http://localhost/api/ai/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: "gemini",
          apiKey: "gemini-test-key",
          model: "gemini-3.7-flash",
          spot: SPOT,
        }),
      }),
    );
    const reader = response.body!.getReader();

    try {
      const first = await Promise.race([
        reader.read(),
        new Promise<"timeout">((resolve) =>
          setTimeout(() => resolve("timeout"), 50),
        ),
      ]);
      expect(first).not.toBe("timeout");
      if (first !== "timeout") {
        expect(new TextDecoder().decode(first.value)).toBe("First token");
      }
    } finally {
      await reader.cancel();
    }
  });

  it("handles a CRLF event boundary split across upstream chunks", async () => {
    const encoder = new TextEncoder();
    const upstream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(
          encoder.encode(
            'data: {"candidates":[{"content":{"parts":[{"text":"Split token"}]}}]}\r',
          ),
        );
        controller.enqueue(encoder.encode("\n\r\n"));
      },
    });
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(upstream, {
          headers: { "Content-Type": "text/event-stream" },
        }),
      ),
    );

    const response = await POST(
      new NextRequest("http://localhost/api/ai/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: "gemini",
          apiKey: "gemini-test-key",
          model: "gemini-3.7-flash",
          spot: SPOT,
        }),
      }),
    );
    const reader = response.body!.getReader();

    try {
      const first = await Promise.race([
        reader.read(),
        new Promise<"timeout">((resolve) =>
          setTimeout(() => resolve("timeout"), 50),
        ),
      ]);
      expect(first).not.toBe("timeout");
      if (first !== "timeout") {
        expect(new TextDecoder().decode(first.value)).toBe("Split token");
      }
    } finally {
      await reader.cancel();
    }
  });

  it("closes after Gemini's terminal event without waiting for the socket", async () => {
    const encoder = new TextEncoder();
    const upstream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(
          encoder.encode(
            'data: {"candidates":[{"content":{"parts":[{"text":"Complete"}]},"finishReason":"STOP"}]}\r\n\r\n',
          ),
        );
      },
    });
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(upstream, {
          headers: { "Content-Type": "text/event-stream" },
        }),
      ),
    );

    const response = await POST(
      new NextRequest("http://localhost/api/ai/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: "gemini",
          apiKey: "gemini-test-key",
          model: "gemini-3.7-flash",
          spot: SPOT,
        }),
      }),
    );
    const reader = response.body!.getReader();

    expect(new TextDecoder().decode((await reader.read()).value)).toBe(
      "Complete",
    );
    const completed = await Promise.race([
      reader.read(),
      new Promise<"timeout">((resolve) =>
        setTimeout(() => resolve("timeout"), 50),
      ),
    ]);
    expect(completed).not.toBe("timeout");
    if (completed !== "timeout") expect(completed.done).toBe(true);
  });
});
