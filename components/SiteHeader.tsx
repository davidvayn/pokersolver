'use client';

import { usePathname } from 'next/navigation';
import { DealerRailHeader } from '@/components/navigation/DealerRailHeader';
import { useUi } from '@/lib/ui-store';

export function SiteHeader() {
  const pathname = usePathname();
  const openSettings = useUi((state) => state.openSettings);

  return (
    <DealerRailHeader
      pathname={pathname}
      onOpenSettings={openSettings}
    />
  );
}
