'use client';

import { useEffect, useRef, useState } from 'react';
import { chatGptConnection, type ChatGptConnection as Connection } from '@/lib/ai/chatgpt';
import type { AiSettings } from '@/lib/ai/settings';

export function ChatGptConnection({
  settings, update,
}: {
  settings: AiSettings;
  update: (next: Partial<AiSettings>) => void;
}) {
  const [connection, setConnection] = useState<Connection | null>(null);
  const [waiting, setWaiting] = useState(false);
  const [error, setError] = useState('');
  const [account, setAccount] = useState(settings.chatgptAccounts?.at(-1)?.clientId ?? 'new');
  const [startedAt, setStartedAt] = useState(0);
  const latest = useRef({ settings, update });
  latest.current = { settings, update };

  function acceptConnection(next: Connection) {
    setConnection(next);
    if (!next.connected || !next.registration) return;
    const { settings: current, update: updateCurrent } = latest.current;
    const accounts = current.chatgptAccounts ?? [];
    const models = next.models ?? [];
    updateCurrent({
      chatgptAccounts: [
        ...accounts.filter((item) => item.clientId !== next.registration!.clientId),
        next.registration,
      ],
      ...(models.length && !models.some((item) => item.id === current.model)
        ? { model: models[0].id } : {}),
    });
  }

  useEffect(() => {
    let active = true;
    chatGptConnection().then((next) => {
      if (active) acceptConnection(next);
    }).catch(() => { if (active) setError('Could not check ChatGPT sign-in. Try again.'); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!waiting) return;
    let active = true;
    let checking = false;
    const timer = window.setInterval(async () => {
      if (checking) return;
      if (Date.now() - startedAt > 10 * 60_000) {
        setWaiting(false);
        setError('Sign-in timed out. Try connecting again.');
        return;
      }
      checking = true;
      try {
        const next = await chatGptConnection();
        if (active && next.connected) {
          acceptConnection(next);
          setWaiting(false);
        }
      } catch { /* Wait for the next status check during authorization. */ }
      finally { checking = false; }
    }, 2000);
    return () => { active = false; window.clearInterval(timer); };
  }, [waiting, startedAt]);

  async function connect() {
    setError('');
    const signInTab = window.open('about:blank', '_blank');
    if (!signInTab) {
      setError('Allow a new tab for ChatGPT sign-in, then try again.');
      return;
    }
    try {
      const hostId = settings.chatgptHostId ?? `urn:uuid:${crypto.randomUUID()}`;
      update({ chatgptHostId: hostId, openaiAuth: 'chatgpt' });
      const response = await fetch('/api/ai/chatgpt', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'connect', hostId,
          registration: settings.chatgptAccounts?.find((item) => item.clientId === account),
        }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Could not start ChatGPT sign-in.');
      signInTab.opener = null;
      signInTab.location.replace(result.authorizationUrl);
      setStartedAt(Date.now());
      setWaiting(true);
    } catch (caught) {
      signInTab.close();
      setError((caught as Error).message);
    }
  }

  async function disconnect() {
    setError('');
    try {
      const response = await fetch('/api/ai/chatgpt', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'disconnect' }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Could not disconnect ChatGPT.');
      setConnection({ available: true, connected: false });
      update({ openaiAuth: 'chatgpt' });
      if (!result.revoked) {
        setError('Signed out locally. Remote disconnection could not be confirmed; remove Poker Lab in ChatGPT Settings.');
      }
    } catch (caught) { setError((caught as Error).message); }
  }

  return (
    <div className="mb-4 space-y-3 text-sm">
      {connection?.connected ? (
        <>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-fg">Connected · {connection.registration?.email ?? 'ChatGPT account'}</span>
            <button type="button" onClick={disconnect} className="min-h-11 text-muted underline focus-visible:ring-2 focus-visible:ring-accent">Disconnect</button>
          </div>
          <label className="block font-medium">
            Model
            <select
              value={settings.model}
              onChange={(event) => update({ model: event.target.value })}
              className="mt-2 min-h-11 w-full rounded-md border border-border bg-surface-2 px-3 py-2 text-base outline-none focus-visible:ring-2 focus-visible:ring-accent"
              disabled={!connection.models?.length}
            >
              {connection.models?.map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}
            </select>
          </label>
          {connection.error && <p role="alert" className="text-muted">{connection.error}</p>}
          <a href="https://chatgpt.com/settings/usage" target="_blank" rel="noreferrer" className="inline-flex min-h-11 items-center text-muted underline focus-visible:ring-2 focus-visible:ring-accent">Plan usage and app access</a>
        </>
      ) : (
        <>
          {!!settings.chatgptAccounts?.length && (
            <label className="block">
              Account
              <select value={account} onChange={(event) => setAccount(event.target.value)} className="mt-2 min-h-11 w-full rounded-md border border-border bg-surface-2 px-3 py-2">
                <option value="new">Use a different ChatGPT account</option>
                {settings.chatgptAccounts.map((item) => <option key={item.clientId} value={item.clientId}>{item.email ?? item.subject} · {item.clientId.slice(-6)}</option>)}
              </select>
            </label>
          )}
          <button
            type="button" onClick={connect} disabled={!connection?.available || waiting}
            className="min-h-11 w-full rounded-md border border-border bg-fg px-3 py-2 font-semibold text-bg disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            {waiting ? 'Finish sign-in in the new tab…' : 'Continue with ChatGPT'}
          </button>
          {waiting && <button type="button" onClick={() => setWaiting(false)} className="min-h-11 text-muted underline">Cancel waiting</button>}
          <p className="text-xs leading-relaxed text-muted">
            {connection?.available === false
              ? 'Website sign-in needs an approved OpenAI OAuth registration. API-key access remains available.'
              : 'Use an eligible ChatGPT plan without an API key. This local connection lasts until the server restarts. Your credentials stay on the local server.'}
          </p>
        </>
      )}
      {error && <p role="alert" className="text-xs text-muted">{error}</p>}
    </div>
  );
}
