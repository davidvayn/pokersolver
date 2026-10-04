import type { Metadata } from 'next';
import './globals.css';
import './redesign.css';
import { DesignShell } from '@/components/design/DesignShell';
import { PracticeModelWarmer } from '@/components/practice/PracticeModelWarmer';
import { ChatGptReturn } from '@/components/ai/ChatGptReturn';

export const metadata: Metadata = {
  title: "Poker Lab - Texas Hold'em Training",
  description:
    'Private preflop practice, performance analysis, range charts, and local postflop solving.',
};

export const viewport = {
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#f7f8fa' },
    { media: '(prefers-color-scheme: dark)', color: '#0c0f14' },
  ],
};

// Set the theme class before paint to avoid a flash of the wrong theme.
const themeInit = `(function(){try{var t=localStorage.getItem('theme');var d=t?t==='dark':window.matchMedia('(prefers-color-scheme: dark)').matches;document.documentElement.classList.toggle('dark',d);}catch(e){document.documentElement.classList.add('dark');}})();`;

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInit }} />
      </head>
      <body className="min-h-full">
        <ChatGptReturn />
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-accent focus:px-4 focus:py-2 focus:text-accent-fg"
        >
          Skip to content
        </a>
        <DesignShell>{children}</DesignShell>
        <PracticeModelWarmer />
      </body>
    </html>
  );
}
