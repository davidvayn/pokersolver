import type { ProviderId } from "./providers";

/** Parse provider SSE and emit only the text deltas as a UTF-8 stream. */
export function normalizeStream(
  body: ReadableStream<Uint8Array>,
  provider: ProviderId | "responses",
): ReadableStream<Uint8Array> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = "";
  let pendingCarriageReturn = false;
  let completed = false;

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
          obj?.type === "response.failed"
            ? obj.response?.error?.message || "ChatGPT could not complete this response."
            : obj?.type === "response.incomplete"
              ? "ChatGPT stopped before completing the response. Please try again."
            : obj?.type === "error"
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
          provider === "responses"
            ? obj.type === "response.output_text.delta" ? obj.delta ?? "" : ""
            : provider === "anthropic"
            ? obj.type === "content_block_delta"
              ? (obj.delta?.text ?? "")
              : ""
            : provider === "gemini"
              ? (obj.candidates?.[0]?.content?.parts ?? [])
                  .map((part: { text?: string }) => part.text ?? "")
                  .join("")
              : (obj.choices?.[0]?.delta?.content ?? "");
        if (text) controller.enqueue(encoder.encode(text));
        if (
          provider === "gemini" &&
          obj.candidates?.[0]?.finishReason === "MAX_TOKENS"
        ) {
          controller.enqueue(
            encoder.encode(
              "\n\nResponse reached the model's output limit. Ask a follow-up to continue.",
            ),
          );
        }
        const streamFinished =
          (provider === "responses" && obj.type === "response.completed") ||
          (provider === "gemini" &&
            Boolean(obj.candidates?.[0]?.finishReason)) ||
          (provider === "anthropic" && obj.type === "message_stop");
        if (streamFinished) {
          completed = true;
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
        if (provider === "responses" && !completed) {
          controller.enqueue(encoder.encode("\n\nChatGPT's connection ended before the response completed. Please try again."));
        }
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
