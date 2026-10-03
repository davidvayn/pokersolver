'use client';

import { useEffect, useState } from 'react';
import { chatGptConnection } from '@/lib/ai/chatgpt';
import { loadSettings, saveSettings } from '@/lib/ai/settings';
import { useUi } from '@/lib/ui-store';

export function ChatGptReturn() {
  const openSettings = useUi((state) => state.openSettings);
  const [error, setError] = useState('');
  useEffect(() => {
    const url = new URL(window.location.href);
    const result = url.searchParams.get('chatgpt');
    if (!result) return;
    url.searchParams.delete('chatgpt');
    window.history.replaceState(window.history.state, '', url);
    async function finish() {
      if (result === 'connected') {
        const connection = await chatGptConnection();
        if (connection.connected && connection.registration) {
          const settings = loadSettings();
          saveSettings({
            ...settings, provider: 'openai', openaiAuth: 'chatgpt',
            model: connection.models?.some((model) => model.id === settings.model)
              ? settings.model : connection.models?.[0]?.id ?? settings.model,
            chatgptAccounts: [
              ...(settings.chatgptAccounts ?? []).filter((account) => account.clientId !== connection.registration!.clientId),
              connection.registration,
            ],
          });
        }
      } else {
        setError(result === 'denied'
          ? 'ChatGPT sign-in was canceled. You can reconnect in Settings.'
          : 'ChatGPT sign-in could not be verified. Please try again in Settings.');
      }
      openSettings();
    }
    void finish().catch(() => {
      setError('Could not load the ChatGPT connection. Please try again in Settings.');
      openSettings();
    });
  }, [openSettings]);
  return error ? (
    <div role="alert" className="fixed bottom-24 left-4 right-4 z-[110] mx-auto flex max-w-md items-center gap-3 rounded-md border border-border bg-surface p-3 text-sm shadow-card">
      <p>{error}</p>
      <button type="button" onClick={() => setError('')} className="min-h-11 shrink-0 text-muted underline">Dismiss</button>
    </div>
  ) : null;
}
