// AI provider definitions. Adding a provider here + a branch in the analyze
// route is all that's needed to support it.

export type ProviderId = 'anthropic' | 'openai' | 'gemini';

export interface ProviderDef {
  id: ProviderId;
  label: string;
  defaultModel: string;
  models: string[];
  keyPlaceholder: string;
  keyUrl: string;
}

export const PROVIDERS: Record<ProviderId, ProviderDef> = {
  anthropic: {
    id: 'anthropic',
    label: 'Anthropic (Claude)',
    defaultModel: 'claude-sonnet-5',
    models: ['claude-sonnet-5', 'claude-opus-4-8', 'claude-haiku-4-5-20251001'],
    keyPlaceholder: 'sk-ant-…',
    keyUrl: 'https://console.anthropic.com/settings/keys',
  },
  openai: {
    id: 'openai',
    label: 'OpenAI (GPT)',
    defaultModel: 'gpt-4o',
    models: ['gpt-4o', 'gpt-4o-mini', 'gpt-4.1'],
    keyPlaceholder: 'sk-…',
    keyUrl: 'https://platform.openai.com/api-keys',
  },
  gemini: {
    id: 'gemini',
    label: 'Google (Gemini)',
    defaultModel: 'gemini-3.7-flash',
    models: [
      'gemini-3.7-flash',
      'gemini-3.6-flash',
      'gemini-3.1-flash-lite',
      'gemini-3.5-flash-lite',
      'gemini-2.5-pro',
    ],
    keyPlaceholder: 'AIza…',
    keyUrl: 'https://aistudio.google.com/app/apikey',
  },
};

export const PROVIDER_LIST = Object.values(PROVIDERS);
