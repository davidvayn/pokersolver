import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  // Next preserves JSX for its own compiler. Vite 8's Oxc transform must
  // compile it for the Node-based component rendering tests instead.
  oxc: {
    jsx: { runtime: 'automatic' },
  },
  resolve: {
    alias: { '@': fileURLToPath(new URL('.', import.meta.url)) },
  },
  test: {
    environment: 'node',
    include: ['**/*.test.ts'],
  },
});
