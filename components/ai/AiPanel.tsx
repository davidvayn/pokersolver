'use client';

import { useState } from 'react';
import Link from 'next/link';
import { loadSettings, currentKey } from '@/lib/ai/settings';
import type { SpotContext } from '@/lib/ai/prompt';

interface AiPanelProps {
  /** Called when the user clicks Analyze; returns the spot to send. */
  getSpot: () => SpotContext | null;
}

export function AiPanel({ getSpot }: AiPanelProps) {
  const [text, setText] = useState('');
  const [status, setStatus] = useState<'idle' | 'streaming' | 'error'>('idle');
  const [error, setError] = useState('');

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
    <div className="rounded-lg border border-border bg-surface p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-sm font-semibold">
          <span className="text-accent" aria-hidden="true">
            ✦
          </span>{' '}
          AI Analysis
        </h3>
        <button
          onClick={analyze}
          disabled={status === 'streaming'}
          className="rounded-md bg-accent px-3 py-1.5 text-xs font-semibold text-accent-fg hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:opacity-50"
        >
          {status === 'streaming' ? 'Analyzing…' : 'Analyze this spot'}
        </button>
      </div>

      {status === 'error' && (
        <div
          role="alert"
          className="rounded-md border border-raise/40 bg-raise/10 p-3 text-xs text-raise"
        >
          {error}{' '}
          {error.includes('Settings') && (
            <Link href="/settings" className="underline">
              Open Settings
            </Link>
          )}
        </div>
      )}

      {text ? (
        <div
          aria-live="polite"
          className="prose-poker whitespace-pre-wrap text-sm leading-relaxed text-fg/90"
        >
          {text}
        </div>
      ) : status !== 'error' ? (
        <p className="text-xs text-muted">
          Get a natural-language read on the current spot. Uses your own API key
          (set in <Link href="/settings" className="underline">Settings</Link>).
        </p>
      ) : null}
    </div>
  );
}
