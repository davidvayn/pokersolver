import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  // Compile JSX for Node-based component rendering tests independently of
  // the application's Next.js transform settings.
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
