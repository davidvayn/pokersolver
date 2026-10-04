'use client';

import { useEffect, type ReactNode } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Settings2, Spade } from 'lucide-react';
import { NAV_ITEMS } from '@/components/navigation/nav-items';
import { SiteHeader } from '@/components/SiteHeader';
import { ThemeToggle } from '@/components/ThemeToggle';
import { SettingsModal } from '@/components/settings/SettingsModal';

export function PageHeading({
  title,
  meta,
  children,
}: {
  title: string;
  meta?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <header className="study-page-heading">
      <div>
        <h1>{title}</h1>
        {meta && <p className="study-page-meta">{meta}</p>}
      </div>
      {children && <div className="study-page-actions">{children}</div>}
    </header>
  );
}

export function DesignShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const solver = pathname === '/solver' || pathname.startsWith('/solver/');

  useEffect(() => {
    // Retire prototype links while preserving other query parameters and hashes.
    const cleanUrl = () => {
      const url = new URL(window.location.href);
      if (!url.searchParams.has('design')) return;
      url.searchParams.delete('design');
      window.history.replaceState(
        null,
        '',
        `${url.pathname}${url.search}${url.hash}`
      );
    };
    cleanUrl();
    window.addEventListener('popstate', cleanUrl);
    return () => window.removeEventListener('popstate', cleanUrl);
  }, [pathname]);

  const nav = (mobile = false) => (
    <nav
      aria-label={mobile ? 'Mobile navigation' : 'Primary navigation'}
      className={mobile ? 'design-mobile-nav' : 'design-nav'}
    >
      {NAV_ITEMS.map(({ href, label, icon: Icon }) => (
        <Link
          key={href}
          href={href}
          aria-current={pathname.startsWith(href) ? 'page' : undefined}
        >
          <Icon className="h-5 w-5" aria-hidden="true" />
          <span>{label}</span>
        </Link>
      ))}
    </nav>
  );

  return (
    <>
      {solver ? (
        <>
          <SiteHeader />
          <main
            id="main"
            className="mx-auto w-full max-w-[1400px] px-4 py-6 pb-[calc(6rem+env(safe-area-inset-bottom))] md:pb-6"
          >
            {children}
          </main>
        </>
      ) : (
        <div className="app-redesign">
          <header className="design-header">
            <Link href="/" className="design-brand" aria-label="Poker Lab home">
              <Spade className="h-6 w-6 text-accent" aria-hidden="true" />
              <span>Poker Lab</span>
            </Link>
            {nav()}
            <div className="design-utilities">
              <Link
                href="/settings"
                aria-label="Settings"
                className="design-icon-button"
              >
                <Settings2 className="h-5 w-5" aria-hidden="true" />
              </Link>
              <ThemeToggle />
            </div>
          </header>
          <main id="main" className="design-main">
            {children}
          </main>
          {nav(true)}
        </div>
      )}
      <SettingsModal />
    </>
  );
}
