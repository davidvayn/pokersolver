import { describe, expect, it } from 'vitest';
import { PROVIDERS, PROVIDER_LIST } from './providers';

describe('AI providers', () => {
  it('offers Gemini with a stable text model and key destination', () => {
    expect(PROVIDERS.gemini.defaultModel).toBe('gemini-3.7-flash');
    expect(PROVIDERS.gemini.models).toContain(PROVIDERS.gemini.defaultModel);
    expect(PROVIDERS.gemini.models).toContain('gemini-3.1-flash-lite');
    expect(PROVIDERS.gemini.keyUrl).toBe(
      'https://aistudio.google.com/app/apikey'
    );
    expect(PROVIDER_LIST.map((provider) => provider.id)).toContain('gemini');
  });
});
