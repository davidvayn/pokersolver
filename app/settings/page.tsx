'use client';

import { useEffect, useState } from 'react';
import { PROVIDER_LIST, PROVIDERS, ProviderId } from '@/lib/ai/providers';
import { AiSettings, loadSettings, saveSettings } from '@/lib/ai/settings';

export default function SettingsPage() {
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
    <div className="mx-auto flex max-w-xl flex-col gap-6">
      <div>
        <h1 className="text-xl font-semibold">Settings</h1>
        <p className="text-sm text-muted">
          Your API key is stored only in this browser and forwarded through a
          serverless proxy to the provider — never persisted on the server.
        </p>
      </div>

      <div className="rounded-lg border border-border bg-surface p-5">
        <label className="mb-2 block text-sm font-medium">AI Provider</label>
        <div className="mb-4 flex flex-wrap gap-2">
          {PROVIDER_LIST.map((p) => (
            <button
              key={p.id}
              onClick={() => setProvider(p.id)}
              className={
                'rounded-md border px-3 py-1.5 text-sm transition-colors ' +
                (settings.provider === p.id
                  ? 'border-accent text-fg'
                  : 'border-border text-muted hover:text-fg')
              }
            >
              {p.label}
            </button>
          ))}
        </div>

        <label className="mb-2 block text-sm font-medium">Model</label>
        <select
          value={settings.model}
          onChange={(e) => update({ model: e.target.value })}
          className="mb-4 w-full rounded-md border border-border bg-surface-2 p-2 text-sm outline-none focus:border-accent"
        >
          {provider.models.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>

        <label className="mb-2 block text-sm font-medium">
          {provider.label} API Key
        </label>
        <input
          type="password"
          value={settings.apiKeys[settings.provider] ?? ''}
          onChange={(e) => setKey(e.target.value)}
          placeholder={provider.keyPlaceholder}
          spellCheck={false}
          className="w-full rounded-md border border-border bg-surface-2 p-2 font-mono text-sm outline-none focus:border-accent"
        />
        <div className="mt-2 flex items-center justify-between text-xs">
          <a
            href={provider.keyUrl}
            target="_blank"
            rel="noreferrer"
            className="text-muted underline hover:text-fg"
          >
            Get a key ↗
          </a>
          <span className={saved ? 'text-accent' : 'text-transparent'}>
            Saved
          </span>
        </div>
      </div>

      <p className="text-xs text-muted">
        Tip: keys never leave your machine except to call the provider you
        selected. Clear the field to remove a stored key.
      </p>
    </div>
  );
}
