'use client';

export interface ChatGptRegistration {
  clientId: string;
  subject: string;
  email?: string;
}
export interface ChatGptConnection {
  available: boolean;
  connected: boolean;
  registration?: ChatGptRegistration;
  models?: { id: string; label: string }[];
  error?: string;
}

export async function chatGptConnection(): Promise<ChatGptConnection> {
  const response = await fetch('/api/ai/chatgpt', { cache: 'no-store' });
  if (!response.ok) throw new Error('Could not check ChatGPT sign-in.');
  return response.json();
}
