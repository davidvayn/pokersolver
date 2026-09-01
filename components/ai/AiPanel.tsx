'use client';

import { useState } from 'react';
import { GeminiMark } from '@/components/ai/GeminiMark';
import { loadSettings, currentKey } from '@/lib/ai/settings';
import { useUi } from '@/lib/ui-store';
import type { SpotContext } from '@/lib/ai/prompt';

interface AiPanelProps {
  /** Called when the user clicks Analyze; returns the spot to send. */
  getSpot: () => SpotContext | null;
  /** Removes the outer card when the panel is mounted inside another surface. */
  embedded?: boolean;
}

export function AiPanel({ getSpot, embedded = false }: AiPanelProps) {
  const [text, setText] = useState('');
  const [status, setStatus] = useState<'idle' | 'streaming' | 'error'>('idle');
  const [error, setError] = useState('');
  const openSettings = useUi((s) => s.openSettings);

  async function analyze() {
    const settings = loadSettings();
    const key = currentKey(settings);
    if (!key) {
      setStatus('error');
      setError('No API key set. Add one in Settings.');
      return;
    }
    const spot = getSpot();
    if (!spot) {
      setStatus('error');
      setError('Nothing to analyze yet — set up a spot first.');
      return;
    }

    setText('');
    setError('');
    setStatus('streaming');
    try {
      const res = await fetch('/api/ai/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: settings.provider,
          apiKey: key,
          model: settings.model,
          spot,
        }),
      });

      if (!res.ok || !res.body) {
        const err = await res.json().catch(() => ({ error: res.statusText }));
        throw new Error(err.error || 'Request failed');
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        setText((t) => t + decoder.decode(value, { stream: true }));
      }
      setStatus('idle');
    } catch (e) {
      setStatus('error');
      setError((e as Error).message);
    }
  }

  return (
    <div
      className={
        embedded
          ? 'flex h-full min-h-0 flex-col'
          : 'flex min-h-0 flex-col rounded-lg border border-border bg-surface p-4'
      }
    >
      <div className="mb-3 flex shrink-0 items-center justify-between gap-3">
        <h3 className="flex items-center gap-2 text-sm font-semibold">
          <GeminiMark className="h-5 w-5" />
          AI Analysis
        </h3>
        <button
          type="button"
          onClick={analyze}
          disabled={status === 'streaming'}
          className="min-h-11 rounded-md bg-accent px-3 py-2 text-xs font-semibold text-accent-fg transition-opacity hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:cursor-not-allowed disabled:opacity-50"
        >
          {status === 'streaming' ? 'Analyzing…' : 'Analyze this spot'}
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto pr-1">
        {status === 'error' && (
          <div
            role="alert"
            className="rounded-md border border-raise/40 bg-raise/10 p-3 text-xs text-raise"
          >
            {error}{' '}
            {error.includes('Settings') && (
              <button type="button" onClick={openSettings} className="underline">
                Open Settings
              </button>
            )}
          </div>
        )}

        {text ? (
          <div
            aria-live="polite"
            className="prose-poker max-w-3xl whitespace-pre-wrap text-sm leading-relaxed text-fg/90"
          >
            {text}
          </div>
        ) : status !== 'error' ? (
          <p className="text-xs text-muted">
            Get a natural-language read on the current spot. Uses your own API
            key (set in{' '}
            <button type="button" onClick={openSettings} className="underline">
              Settings
            </button>
            ).
          </p>
        ) : null}
      </div>
    </div>
  );
}
