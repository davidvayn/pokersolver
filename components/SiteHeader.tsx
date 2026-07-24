'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  BarChart3,
  BrainCircuit,
  Grid3X3,
  Settings,
  Spade,
  Target,
} from 'lucide-react';
import { ThemeToggle } from '@/components/ThemeToggle';
import { useUi } from '@/lib/ui-store';

const NAV = [
  { href: '/preflop', label: 'Preflop', icon: Grid3X3 },
  { href: '/practice', label: 'Practice', icon: Target },
  { href: '/solver', label: 'Solver', icon: BrainCircuit },
  { href: '/stats', label: 'Stats', icon: BarChart3 },
];

export function SiteHeader() {
  const pathname = usePathname();
  const openSettings = useUi((state) => state.openSettings);

  return (
    <>
      <header className="sticky top-0 z-40 border-b border-border bg-surface/90 backdrop-blur">
        <div className="mx-auto flex h-14 w-full max-w-[1400px] items-center gap-6 px-4">
          <Link
            href="/"
            className="flex min-h-11 items-center gap-2 font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            <span
              className="grid h-8 w-8 place-items-center rounded-md bg-felt text-accent-fg"
              aria-hidden
            >
              <Spade className="h-4 w-4" />
            </span>
            <span>Poker Lab</span>
          </Link>

          <nav className="hidden items-center gap-1 text-sm md:flex">
            {NAV.map((item) => {
              const active =
                pathname === item.href || pathname.startsWith(`${item.href}/`);
              const Icon = item.icon;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  aria-current={active ? 'page' : undefined}
                  className={
                    'flex min-h-11 items-center gap-2 rounded-md px-3 py-2 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ' +
                    (active
                      ? 'bg-surface-2 text-fg'
                      : 'text-muted hover:bg-surface-2 hover:text-fg')
                  }
                >
                  <Icon className="h-4 w-4" aria-hidden="true" />
                  {item.label}
                </Link>
              );
            })}
          </nav>

          <div className="ml-auto flex items-center gap-2">
            <button
              type="button"
              onClick={(event) => {
                event.currentTarget.focus();
                openSettings();
              }}
              aria-label="Open settings"
              title="Settings"
              className="grid h-11 w-11 place-items-center rounded-md border border-border text-muted transition-colors hover:bg-surface-2 hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            >
              <Settings className="h-4 w-4" aria-hidden="true" />
            </button>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <nav
        aria-label="Primary navigation"
        className="fixed inset-x-0 bottom-0 z-40 grid min-h-[calc(4rem+env(safe-area-inset-bottom))] grid-cols-4 border-t border-border bg-surface/95 px-1 pb-[env(safe-area-inset-bottom)] backdrop-blur md:hidden"
      >
        {NAV.map((item) => {
          const active =
            pathname === item.href || pathname.startsWith(`${item.href}/`);
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              aria-current={active ? 'page' : undefined}
              className={
                'flex min-w-0 flex-col items-center justify-center gap-1 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent ' +
                (active ? 'text-accent' : 'text-muted hover:text-fg')
              }
            >
              <Icon className="h-5 w-5" aria-hidden="true" />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>
    </>
  );
}
