import type { Config } from 'tailwindcss';

const config: Config = {
  darkMode: 'class',
  content: [
    './app/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './lib/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        bg: 'rgb(var(--bg) / <alpha-value>)',
        surface: 'rgb(var(--surface) / <alpha-value>)',
        'surface-2': 'rgb(var(--surface-2) / <alpha-value>)',
        border: 'rgb(var(--border) / <alpha-value>)',
        fg: 'rgb(var(--fg) / <alpha-value>)',
        muted: 'rgb(var(--muted) / <alpha-value>)',
        accent: 'rgb(var(--accent) / <alpha-value>)',
        'accent-fg': 'rgb(var(--accent-fg) / <alpha-value>)',
        felt: 'rgb(var(--felt) / <alpha-value>)',
        // Poker action colors (used by the hand matrix mixed-strategy bars)
        raise: 'rgb(var(--raise) / <alpha-value>)',
        'raise-fg': 'rgb(var(--raise-fg) / <alpha-value>)',
        call: 'rgb(var(--call) / <alpha-value>)',
        'call-fg': 'rgb(var(--call-fg) / <alpha-value>)',
        fold: 'rgb(var(--fold) / <alpha-value>)',
        'fold-fg': 'rgb(var(--fold-fg) / <alpha-value>)',
        check: 'rgb(var(--check) / <alpha-value>)',
        allin: 'rgb(var(--allin) / <alpha-value>)',
        'allin-fg': 'rgb(var(--allin-fg) / <alpha-value>)',
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      boxShadow: {
        card: '0 1px 3px rgb(0 0 0 / 0.12), 0 1px 2px rgb(0 0 0 / 0.24)',
      },
    },
  },
  plugins: [],
};

export default config;
