import {
  BarChart3,
  BrainCircuit,
  Grid3X3,
  Target,
  type LucideIcon,
} from 'lucide-react';

export type NavItem = {
  href: string;
  label: string;
  icon: LucideIcon;
  shortCode: string;
};

export const NAV_ITEMS: NavItem[] = [
  {
    href: '/preflop',
    label: 'Preflop',
    icon: Grid3X3,
    shortCode: 'PF',
  },
  {
    href: '/practice',
    label: 'Practice',
    icon: Target,
    shortCode: 'DR',
  },
  {
    href: '/solver',
    label: 'Solver',
    icon: BrainCircuit,
    shortCode: 'SV',
  },
  {
    href: '/stats',
    label: 'Stats',
    icon: BarChart3,
    shortCode: 'ST',
  },
];
