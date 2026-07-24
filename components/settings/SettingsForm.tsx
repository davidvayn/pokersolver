'use client';

import { useEffect, useState } from 'react';
import { ExternalLink } from 'lucide-react';
import {
  PROVIDER_LIST,
  PROVIDERS,
  type ProviderId,
} from '@/lib/ai/providers';
import {
  type AiSettings,
  loadSettings,
  saveSettings,
} from '@/lib/ai/settings';

export function SettingsForm() {
  const [settings, setSettings] = useState<AiSettings | null>(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setSettings(loadSettings());
  }, []);

  if (!settings) return null;

  const provider = PROVIDERS[settings.provider];

  function update(next: Partial<AiSettings>) {
    const merged = { ...settings!, ...next };
    setSettings(merged);
    saveSettings(merged);
    setSaved(true);
    setTimeout(() => setSaved(false), 1200);
  }

  function setProvider(id: ProviderId) {
    update({ provider: id, model: PROVIDERS[id].defaultModel });
  }

  function setKey(value: string) {
    update({ apiKeys: { ...settings!.apiKeys, [settings!.provider]: value } });
  }

  return (
    <div>
      <div
        id="provider-label"
        className="mb-2 block text-sm font-medium"
      >
        AI Provider
      </div>
      <div
        role="group"
        aria-labelledby="provider-label"
        className="mb-4 flex flex-wrap gap-2"
      >
        {PROVIDER_LIST.map((p) => (
          <button
            key={p.id}
            type="button"
            onClick={() => setProvider(p.id)}
            aria-pressed={settings.provider === p.id}
            className={
              'min-h-11 rounded-md border px-3 py-2 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ' +
              (settings.provider === p.id
                ? 'border-accent text-fg'
                : 'border-border text-muted hover:text-fg')
            }
          >
            {p.label}
          </button>
        ))}
      </div>

      <label htmlFor="ai-model" className="mb-2 block text-sm font-medium">
        Model
      </label>
      <select
        id="ai-model"
        value={settings.model}
        onChange={(e) => update({ model: e.target.value })}
        className="mb-4 min-h-11 w-full rounded-md border border-border bg-surface-2 px-3 py-2 text-base text-fg outline-none focus:border-accent focus-visible:ring-2 focus-visible:ring-accent"
      >
        {provider.models.map((m) => (
          <option key={m} value={m}>
            {m}
          </option>
        ))}
      </select>

      <label htmlFor="ai-key" className="mb-2 block text-sm font-medium">
        {provider.label} API Key
      </label>
      <input
        id="ai-key"
        name="ai-api-key"
        type="password"
        autoComplete="off"
        value={settings.apiKeys[settings.provider] ?? ''}
        onChange={(e) => setKey(e.target.value)}
        placeholder={provider.keyPlaceholder}
        spellCheck={false}
        className="min-h-11 w-full rounded-md border border-border bg-surface-2 px-3 py-2 font-mono text-base outline-none focus:border-accent focus-visible:ring-2 focus-visible:ring-accent"
      />
      <div className="mt-2 flex items-center justify-between text-xs">
        <a
          href={provider.keyUrl}
          target="_blank"
          rel="noreferrer"
          className="inline-flex min-h-11 items-center gap-1 text-muted underline hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          Get a key
          <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
        </a>
        <span role="status" aria-live="polite" className="text-accent">
          {saved ? 'Saved' : ''}
        </span>
      </div>

      <p className="mt-4 text-xs text-muted">
        Your key is stored only in this browser and forwarded through a
        serverless proxy to the provider. It is never persisted on the server.
      </p>
    </div>
  );
}
