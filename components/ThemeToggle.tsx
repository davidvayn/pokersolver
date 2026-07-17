'use client';

import { useEffect, useState } from 'react';

export function ThemeToggle() {
  const [dark, setDark] = useState(true);

  useEffect(() => {
    setDark(document.documentElement.classList.contains('dark'));
  }, []);

  function toggle() {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle('dark', next);
    try {
      localStorage.setItem('theme', next ? 'dark' : 'light');
    } catch {
      /* ignore */
    }
  }

  return (
    <button
      onClick={toggle}
      className="grid h-8 w-8 place-items-center rounded-md border border-border text-muted hover:text-fg"
      aria-label="Toggle theme"
      title="Toggle theme"
    >
      {dark ? '☾' : '☀'}
    </button>
  );
}
