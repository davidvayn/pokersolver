'use client';

import { useId } from 'react';

export function GeminiMark({ className = 'h-5 w-5' }: { className?: string }) {
  const gradientId = useId();

  return (
    <svg
      viewBox="0 0 24 24"
      className={className}
      aria-hidden="true"
      focusable="false"
    >
      <defs>
        <linearGradient id={gradientId} x1="3" y1="3" x2="21" y2="21">
          <stop offset="0" stopColor="#4285f4" />
          <stop offset="0.52" stopColor="#9b72cb" />
          <stop offset="1" stopColor="#d96570" />
        </linearGradient>
      </defs>
      <path
        fill={`url(#${gradientId})`}
        d="M12 2c.62 5.57 4.43 9.38 10 10-5.57.62-9.38 4.43-10 10-.62-5.57-4.43-9.38-10-10 5.57-.62 9.38-4.43 10-10Z"
      />
    </svg>
  );
}
