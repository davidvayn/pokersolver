'use client';

import Link from 'next/link';
import { Settings, Spade } from 'lucide-react';
import { ThemeToggle } from '@/components/ThemeToggle';
import { NAV_ITEMS } from '@/components/navigation/nav-items';

type DealerRailHeaderProps = {
  pathname: string;
  onOpenSettings: () => void;
};

export function navItemIsActive(pathname: string, href: string) {
  return pathname === href || pathname.startsWith(`${href}/`);
}

export function HeaderHomeMark({ className = '' }: { className?: string }) {
  return (
    <Link
      href="/"
      aria-label="Poker Lab home"
      className={`grid h-11 w-11 shrink-0 place-items-center rounded-md bg-felt text-accent-fg transition-opacity hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg ${className}`}
    >
      <Spade className="h-5 w-5" aria-hidden="true" />
    </Link>
  );
}

export function HeaderUtilities({
  onOpenSettings,
}: {
  onOpenSettings: () => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <button
        type="button"
        onClick={(event) => {
          event.currentTarget.focus();
          onOpenSettings();
        }}
        aria-label="Open settings"
        title="Settings"
        className="grid h-11 w-11 place-items-center rounded-md border border-border text-muted transition-colors hover:bg-surface-2 hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
      >
        <Settings className="h-4 w-4" aria-hidden="true" />
      </button>
      <ThemeToggle />
    </div>
  );
}

export function DealerRailMobileNav({ pathname }: { pathname: string }) {
  return (
    <nav
      aria-label="Primary navigation"
      className="fixed inset-x-0 bottom-0 z-40 grid min-h-[calc(4.25rem+env(safe-area-inset-bottom))] grid-cols-4 divide-x divide-border border-t border-border bg-surface/95 pb-[env(safe-area-inset-bottom)] font-sans backdrop-blur md:hidden"
    >
      {NAV_ITEMS.map((item, index) => {
        const active = navItemIsActive(pathname, item.href);
        const Icon = item.icon;

        return (
          <Link
            key={item.href}
            href={item.href}
            aria-current={active ? 'page' : undefined}
            className={`relative flex min-w-0 flex-col items-center justify-center gap-1 px-1 text-xs font-semibold leading-4 transition-colors after:absolute after:inset-x-0 after:bottom-0 after:h-0.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent ${
              active
                ? 'bg-surface-2 text-accent after:bg-accent'
                : 'text-muted after:bg-transparent hover:bg-surface-2 hover:text-fg'
            }`}
          >
            <span
              className="absolute right-2 top-1.5 font-mono text-[10px] font-medium leading-none text-muted [font-variant-numeric:tabular-nums]"
              aria-hidden="true"
            >
              0{index + 1}
            </span>
            <Icon className="h-5 w-5" aria-hidden="true" />
            <span>{item.label}</span>
          </Link>
        );
      })}
    </nav>
  );
}

export function DealerRailHeader({
  pathname,
  onOpenSettings,
}: DealerRailHeaderProps) {
  return (
    <>
      <header className="sticky top-0 z-40 border-b border-border bg-bg/95 backdrop-blur">
        <div className="mx-auto flex h-16 w-full max-w-[1600px] items-stretch px-4 font-sans sm:px-8">
          <div className="flex w-16 shrink-0 items-center border-r border-border">
            <HeaderHomeMark />
          </div>
          <nav
            aria-label="Primary navigation"
            className="hidden flex-1 md:grid md:grid-cols-4"
          >
            {NAV_ITEMS.map((item, index) => {
              const active = navItemIsActive(pathname, item.href);

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  aria-current={active ? 'page' : undefined}
                  className={`group relative flex min-h-11 items-center justify-between border-r border-border px-4 text-sm font-semibold leading-5 transition-colors after:absolute after:inset-x-0 after:bottom-0 after:h-0.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent ${
                    active
                      ? 'bg-surface-2 text-fg after:bg-accent'
                      : 'text-muted after:bg-transparent hover:bg-surface-2 hover:text-fg'
                  }`}
                >
                  <span>{item.label}</span>
                  <span
                    className={`font-mono text-xs font-medium leading-none [font-variant-numeric:tabular-nums] ${
                      active
                        ? 'text-accent'
                        : 'text-muted group-hover:text-accent'
                    }`}
                    aria-hidden="true"
                  >
                    0{index + 1}
                  </span>
                </Link>
              );
            })}
          </nav>
          <div className="ml-auto flex w-28 shrink-0 items-center justify-end pl-4">
            <HeaderUtilities onOpenSettings={onOpenSettings} />
          </div>
        </div>
      </header>
      <DealerRailMobileNav pathname={pathname} />
    </>
  );
}
