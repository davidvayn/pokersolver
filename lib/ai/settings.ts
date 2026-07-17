'use client';

import { ProviderId, PROVIDERS } from './providers';

// AI settings live only in the browser's localStorage. The key is sent to our
// serverless proxy per-request and never persisted server-side.

const KEY = 'poker-ai-settings';

export interface AiSettings {
  provider: ProviderId;
  model: string;
  apiKeys: Partial<Record<ProviderId, string>>;
}

export function loadSettings(): AiSettings {
  const fallback: AiSettings = {
    provider: 'anthropic',
    model: PROVIDERS.anthropic.defaultModel,
    apiKeys: {},
  };
  if (typeof window === 'undefined') return fallback;
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return fallback;
    return { ...fallback, ...JSON.parse(raw) };
  } catch {
    return fallback;
  }
}

export function saveSettings(s: AiSettings) {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    /* ignore */
  }
}

export function currentKey(s: AiSettings): string | undefined {
  return s.apiKeys[s.provider];
}
