"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import { SendHorizontal } from "lucide-react";
import { AiMarkdown } from "@/components/ai/AiMarkdown";
import { GeminiMark } from "@/components/ai/GeminiMark";
import {
  buildSpotThreadKey,
  describeSpot,
  type AiConversationMessage,
  type SpotContext,
} from "@/lib/ai/prompt";
import { loadSettings, currentKey } from "@/lib/ai/settings";
import { useUi } from "@/lib/ui-store";

interface AiPanelProps {
  /** Returns the current spot, both for analysis and thread identity. */
  getSpot: () => SpotContext | null;
  /** Removes the outer card when the panel is mounted inside another surface. */
  embedded?: boolean;
}

interface ChatMessage extends AiConversationMessage {
  id: string;
}

interface ArchivedThread {
  label: string;
  analysis: string;
  messages: ChatMessage[];
}

type AiStatus = "idle" | "streaming" | "switching";
type StreamOutcome = "completed" | "aborted" | "failed";

const FIRST_TOKEN_TIMEOUT_MS = 30_000;
const STREAM_IDLE_TIMEOUT_MS = 15_000;

const SUGGESTED_QUESTIONS = [
  "Why this sizing?",
  "Which hands bluff?",
  "What changes on the turn?",
];

export function AiPanel({ getSpot, embedded = false }: AiPanelProps) {
  const currentSpot = getSpot();
  const currentSpotKey = currentSpot ? buildSpotThreadKey(currentSpot) : null;
  const currentSpotLabel = describeSpot(currentSpot);
  const [analysis, setAnalysis] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [status, setStatus] = useState<AiStatus>("idle");
  const [error, setError] = useState("");
  const [threadSpotKey, setThreadSpotKey] = useState<string | null>(null);
  const [threadSpotLabel, setThreadSpotLabel] = useState("");
  const [archivedThreads, setArchivedThreads] = useState<ArchivedThread[]>([]);
  const [newSpotReady, setNewSpotReady] = useState(false);
  const [pendingReplyId, setPendingReplyId] = useState<string | null>(null);
  const requestRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const analysisRef = useRef(analysis);
  const messagesRef = useRef(messages);
  const threadLabelRef = useRef(threadSpotLabel);
  const messageIdRef = useRef(0);
  const openSettings = useUi((state) => state.openSettings);

  analysisRef.current = analysis;
  messagesRef.current = messages;
  threadLabelRef.current = threadSpotLabel;

  useEffect(() => {
    if (!currentSpotKey) return;

    if (!threadSpotKey) {
      setThreadSpotKey(currentSpotKey);
      setThreadSpotLabel(currentSpotLabel);
      return;
    }

    if (currentSpotKey === threadSpotKey) {
      if (status === "switching") setStatus("idle");
      return;
    }

    const hasDiscussion =
      Boolean(analysisRef.current) || messagesRef.current.length > 0;
    if (!hasDiscussion && status !== "streaming") {
      setThreadSpotKey(currentSpotKey);
      setThreadSpotLabel(currentSpotLabel);
      setNewSpotReady(false);
      setError("");
      setStatus("idle");
      return;
    }

    requestRef.current?.abort();
    setStatus("switching");
    setError("");

    const timeout = window.setTimeout(() => {
      const previousAnalysis = analysisRef.current;
      const previousMessages = messagesRef.current;
      if (previousAnalysis || previousMessages.length > 0) {
        setArchivedThreads((threads) =>
          [
            {
              label: threadLabelRef.current,
              analysis: previousAnalysis,
              messages: previousMessages,
            },
            ...threads,
          ].slice(0, 3),
        );
      }
      setAnalysis("");
      setMessages([]);
      setDraft("");
      setPendingReplyId(null);
      setThreadSpotKey(currentSpotKey);
      setThreadSpotLabel(currentSpotLabel);
      setNewSpotReady(true);
      setStatus("idle");
    }, 800);

    return () => window.clearTimeout(timeout);
  }, [currentSpotKey, currentSpotLabel, status, threadSpotKey]);

  useEffect(() => {
    const viewport = scrollRef.current;
    if (viewport) viewport.scrollTop = viewport.scrollHeight;
  }, [analysis, messages, status]);

  useEffect(
    () => () => {
      requestRef.current?.abort();
    },
    [],
  );

  function nextMessageId() {
    messageIdRef.current += 1;
    return `ai-message-${messageIdRef.current}`;
  }

  async function streamResponse(
    spot: SpotContext,
    conversation: AiConversationMessage[],
    onChunk: (chunk: string) => void,
  ): Promise<StreamOutcome> {
    const settings = loadSettings();
    const key = currentKey(settings);
    if (!key) {
      setStatus("idle");
      setError("No API key set. Add one in Settings.");
      return "failed";
    }

    const controller = new AbortController();
    requestRef.current?.abort();
    requestRef.current = controller;
    let receivedText = false;
    let timedOut = false;
    let completedAfterIdle = false;
    let streamIdleTimeout: number | undefined;
    const firstTokenTimeout = window.setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, FIRST_TOKEN_TIMEOUT_MS);
    const armStreamIdleTimeout = () => {
      if (streamIdleTimeout) window.clearTimeout(streamIdleTimeout);
      streamIdleTimeout = window.setTimeout(() => {
        completedAfterIdle = true;
        controller.abort();
      }, STREAM_IDLE_TIMEOUT_MS);
    };
    setError("");
    setStatus("streaming");

    try {
      const response = await fetch("/api/ai/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          provider: settings.provider,
          apiKey: key,
          model: settings.model,
          spot,
          messages: conversation,
        }),
      });

      if (!response.ok || !response.body) {
        const responseError = await response
          .json()
          .catch(() => ({ error: response.statusText }));
        throw new Error(responseError.error || "Request failed");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        if (chunk) {
          if (!receivedText) window.clearTimeout(firstTokenTimeout);
          receivedText = true;
          onChunk(chunk);
          armStreamIdleTimeout();
        }
      }
      const remainder = decoder.decode();
      if (remainder) {
        if (!receivedText) window.clearTimeout(firstTokenTimeout);
        receivedText = true;
        onChunk(remainder);
      }
      if (!receivedText) {
        throw new Error("The AI returned an empty response. Try again.");
      }
      if (requestRef.current === controller) setStatus("idle");
      return "completed";
    } catch (caught) {
      if (completedAfterIdle && receivedText) {
        if (requestRef.current === controller) setStatus("idle");
        return "completed";
      }
      if (timedOut) {
        if (requestRef.current === controller) {
          setStatus("idle");
          setError("AI didn't start responding within 30 seconds. Try again.");
        }
        return "failed";
      }
      if (controller.signal.aborted) return "aborted";
      if (requestRef.current === controller) {
        setStatus("idle");
        setError((caught as Error).message);
      }
      return "failed";
    } finally {
      window.clearTimeout(firstTokenTimeout);
      if (streamIdleTimeout) window.clearTimeout(streamIdleTimeout);
      if (requestRef.current === controller) requestRef.current = null;
    }
  }

  function cancelRequest() {
    const request = requestRef.current;
    if (!request) return;
    requestRef.current = null;
    request.abort();
    setPendingReplyId(null);
    setStatus("idle");
    setError("Response canceled. You can try again when ready.");
  }

  async function analyze() {
    const spot = getSpot();
    if (!spot) {
      setStatus("idle");
      setError("Nothing to analyze yet — set up a spot first.");
      return;
    }

    const key = currentKey(loadSettings());
    if (!key) {
      setStatus("idle");
      setError("No API key set. Add one in Settings.");
      return;
    }

    setThreadSpotKey(buildSpotThreadKey(spot));
    setThreadSpotLabel(describeSpot(spot));
    setAnalysis("");
    setMessages([]);
    setNewSpotReady(false);
    await streamResponse(spot, [], (chunk) => {
      setAnalysis((text) => text + chunk);
    });
  }

  async function sendQuestion(question: string) {
    const content = question.trim();
    const spot = getSpot();
    if (!content || !spot || !analysis || status !== "idle") return;

    const userMessage: ChatMessage = {
      id: nextMessageId(),
      role: "user",
      content,
    };
    const replyId = nextMessageId();
    const replyMessage: ChatMessage = {
      id: replyId,
      role: "assistant",
      content: "",
    };
    const conversation: AiConversationMessage[] = [
      { role: "assistant", content: analysis },
      ...messages.map(({ role, content: messageContent }) => ({
        role,
        content: messageContent,
      })),
      { role: "user", content },
    ];

    setDraft("");
    setPendingReplyId(replyId);
    setMessages((items) => [...items, userMessage, replyMessage]);
    const outcome = await streamResponse(spot, conversation, (chunk) => {
      setMessages((items) =>
        items.map((message) =>
          message.id === replyId
            ? { ...message, content: message.content + chunk }
            : message,
        ),
      );
    });
    if (outcome !== "completed") {
      setMessages((items) =>
        items.filter(
          (message) => message.id !== replyId || Boolean(message.content),
        ),
      );
    }
    setPendingReplyId(null);
  }

  function submitQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void sendQuestion(draft);
  }

  const actionLabel =
    status === "streaming"
      ? "Cancel"
      : status === "switching"
        ? "New thread…"
        : analysis
          ? "Reanalyze"
          : newSpotReady
            ? "Analyze new spot"
            : "Analyze this spot";

  return (
    <div
      className={
        embedded
          ? "flex h-full min-h-0 flex-col"
          : "flex min-h-0 flex-col rounded-lg border border-border bg-surface p-4"
      }
    >
      <div className="mb-2 flex shrink-0 items-center justify-between gap-3">
        <div className="min-w-0">
          <h3 className="flex items-center gap-2 text-sm font-semibold">
            <GeminiMark className="h-5 w-5" />
            AI Analysis
          </h3>
          {threadSpotLabel && (
            <p className="mt-0.5 truncate text-[11px] text-muted">
              {threadSpotLabel}
            </p>
          )}
        </div>
        <button
          type="button"
          onClick={
            status === "streaming" ? cancelRequest : () => void analyze()
          }
          disabled={status === "switching"}
          aria-label={
            status === "streaming" ? "Cancel AI response" : actionLabel
          }
          className={`min-h-11 shrink-0 rounded-md px-3 py-2 text-xs font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:cursor-not-allowed disabled:opacity-50 ${
            status === "streaming"
              ? "border border-border bg-surface text-fg hover:bg-surface-2"
              : "bg-accent text-accent-fg hover:opacity-90"
          }`}
        >
          {actionLabel}
        </button>
      </div>

      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto pr-1">
        {archivedThreads.map((thread, index) => (
          <details
            key={`${thread.label}-${index}`}
            className="mb-2 rounded-md border border-border bg-surface/60 text-xs"
          >
            <summary className="cursor-pointer px-3 py-2 font-medium text-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent">
              Previous spot · {thread.label}
            </summary>
            <div className="max-h-52 space-y-2 overflow-y-auto border-t border-border p-3 text-fg/80">
              <AiMarkdown content={thread.analysis} />
              {thread.messages.map((message) => (
                <div
                  key={message.id}
                  className={`rounded-md px-2.5 py-2 leading-relaxed ${
                    message.role === "user"
                      ? "ml-6 bg-accent/15"
                      : "mr-6 bg-surface-2"
                  }`}
                >
                  {message.role === "assistant" ? (
                    <AiMarkdown content={message.content} />
                  ) : (
                    <p className="whitespace-pre-wrap">{message.content}</p>
                  )}
                </div>
              ))}
            </div>
          </details>
        ))}

        {status === "switching" && (
          <div
            role="status"
            aria-live="polite"
            className="flex items-start gap-3 rounded-md border border-accent/35 bg-accent/10 p-3"
          >
            <GeminiMark className="mt-0.5 h-5 w-5 shrink-0 animate-pulse" />
            <div>
              <p className="text-sm font-semibold">
                Thinking about the new spot…
              </p>
              <p className="mt-1 text-xs leading-relaxed text-muted">
                Saving {threadSpotLabel || "the previous spot"} and preparing{" "}
                {currentSpotLabel}.
              </p>
            </div>
          </div>
        )}

        {newSpotReady && status !== "switching" && !analysis && (
          <div
            role="status"
            className="mb-2 flex items-center gap-2 rounded-md border border-border bg-surface-2/60 px-3 py-2 text-xs"
          >
            <GeminiMark className="h-4 w-4 shrink-0" />
            <span>
              <strong>Fresh thread.</strong>{" "}
              <span className="text-muted">Ready for {threadSpotLabel}.</span>
            </span>
          </div>
        )}

        {error && status !== "switching" && (
          <div
            role="alert"
            className="mb-2 rounded-md border border-raise/40 bg-raise/10 p-3 text-xs text-raise"
          >
            {error}{" "}
            {error.includes("Settings") && (
              <button
                type="button"
                onClick={openSettings}
                className="underline"
              >
                Open Settings
              </button>
            )}
          </div>
        )}

        {status === "streaming" && !analysis && (
          <div
            role="status"
            aria-live="polite"
            className="mb-2 flex items-start gap-3 rounded-md border border-accent/35 bg-accent/10 p-3"
          >
            <GeminiMark className="mt-0.5 h-5 w-5 shrink-0 animate-pulse" />
            <div>
              <p className="text-sm font-semibold">Analyzing this spot…</p>
              <p className="mt-1 text-xs text-muted">
                The response will appear here. You can cancel anytime.
              </p>
            </div>
          </div>
        )}

        {analysis ? (
          <div className="space-y-3 pb-3">
            <div aria-live="polite">
              <AiMarkdown
                content={analysis}
                className="prose-poker text-sm text-fg/90"
              />
            </div>
            {messages.map((message) => (
              <div
                key={message.id}
                className={`rounded-lg px-3 py-2.5 text-sm leading-relaxed ${
                  message.role === "user"
                    ? "ml-8 bg-accent/15 text-fg"
                    : "mr-8 border border-border bg-surface text-fg/90"
                }`}
              >
                {message.content ? (
                  message.role === "assistant" ? (
                    <AiMarkdown content={message.content} />
                  ) : (
                    <p className="whitespace-pre-wrap">{message.content}</p>
                  )
                ) : pendingReplyId === message.id ? (
                  <span
                    className="inline-flex gap-1"
                    aria-label="AI is replying"
                  >
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-muted" />
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-muted [animation-delay:120ms]" />
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-muted [animation-delay:240ms]" />
                  </span>
                ) : null}
              </div>
            ))}
            {messages.length === 0 && status !== "streaming" && (
              <div
                className="flex flex-wrap gap-1.5"
                aria-label="Suggested follow-up questions"
              >
                {SUGGESTED_QUESTIONS.map((question) => (
                  <button
                    key={question}
                    type="button"
                    onClick={() => void sendQuestion(question)}
                    className="min-h-9 rounded-full border border-border bg-surface px-3 text-[11px] text-muted transition-colors hover:border-accent hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                  >
                    {question}
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : status === "idle" && !newSpotReady ? (
          <p className="text-xs leading-relaxed text-muted">
            Analyze this spot, then ask follow-up questions without leaving the
            solver. Uses your API key from{" "}
            <button type="button" onClick={openSettings} className="underline">
              Settings
            </button>
            .
          </p>
        ) : null}
      </div>

      {analysis && (
        <form
          onSubmit={submitQuestion}
          className="mt-2 flex shrink-0 items-center gap-2 border-t border-border pt-2"
        >
          <label htmlFor="ai-follow-up" className="sr-only">
            Ask a follow-up question
          </label>
          <input
            id="ai-follow-up"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            disabled={status !== "idle"}
            placeholder="Ask about this spot…"
            autoComplete="off"
            className="min-h-11 min-w-0 flex-1 rounded-md border border-border bg-surface px-3 text-sm text-fg placeholder:text-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={!draft.trim() || status !== "idle"}
            aria-label="Send follow-up"
            className="grid h-11 w-11 shrink-0 place-items-center rounded-md bg-accent text-accent-fg transition-opacity hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:cursor-not-allowed disabled:opacity-40"
          >
            <SendHorizontal className="h-4 w-4" aria-hidden="true" />
          </button>
        </form>
      )}
    </div>
  );
}
