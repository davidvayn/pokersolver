'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ThemeToggle } from '@/components/ThemeToggle';

const NAV = [
  { href: '/calculator', label: 'Equity' },
  { href: '/charts', label: 'Preflop Charts' },
  { href: '/solver', label: 'Postflop Solver' },
  { href: '/settings', label: 'Settings' },
];

export function SiteHeader() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-40 border-b border-border bg-surface/80 backdrop-blur">
      <div className="mx-auto flex h-14 w-full max-w-[1400px] items-center gap-6 px-4">
        <Link href="/" className="flex items-center gap-2 font-semibold">
          <span
            className="grid h-7 w-7 place-items-center rounded-md text-accent-fg"
            style={{ background: 'rgb(var(--felt))' }}
            aria-hidden
          >
            ♠
          </span>
          <span className="tracking-tight">Solver</span>
        </Link>
        <nav className="flex items-center gap-1 text-sm">
          {NAV.map((item) => {
            const active =
              pathname === item.href || pathname.startsWith(item.href + '/');
            return (
              <Link
                key={item.href}
                href={item.href}
                className={
                  'rounded-md px-3 py-1.5 transition-colors ' +
                  (active
                    ? 'bg-surface-2 text-fg'
                    : 'text-muted hover:bg-surface-2 hover:text-fg')
                }
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="ml-auto">
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
