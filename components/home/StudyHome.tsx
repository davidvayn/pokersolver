'use client';

import {
  ArrowRight,
  BarChart3,
  BrainCircuit,
  Grid3X3,
  Target,
} from 'lucide-react';
import Link from 'next/link';
import {
  HandMatrix,
  type StrategySegment,
} from '@/components/hand-matrix/HandMatrix';
import { chartToStrategy } from '@/lib/preflop';
import { scenariosForSeats } from '@/data/preflop/catalog';

const scenario = scenariosForSeats(6)[0];
const chart = scenario?.charts.find(
  (entry) => entry.hero === 'BTN' && entry.category === 'RFI'
);
const strategy: Record<string, StrategySegment[]> = chart
  ? chartToStrategy(chart)
  : {};
const destinations = [
  {
    href: '/practice',
    title: 'Practice',
    detail: 'Play. Decide. Review.',
    icon: Target,
  },
  {
    href: '/preflop',
    title: 'Ranges',
    detail: 'Every seat. One matrix.',
    icon: Grid3X3,
  },
  {
    href: '/solver',
    title: 'Solver',
    detail: 'Build a spot. Find the mix.',
    icon: BrainCircuit,
  },
  {
    href: '/stats',
    title: 'Your game',
    detail: 'Track decisions. Find leaks.',
    icon: BarChart3,
  },
];

export function StudyHome() {
  return (
    <div className="study-home">
      <section className="home-intro" aria-labelledby="home-title">
        <span className="study-kicker">Your study table</span>
        <h1 id="home-title">
          Study the spot.
          <br />
          Own the decision.
        </h1>
        <p>Ranges, practice, and analysis. All in one place.</p>
        <Link href="/practice" className="study-button-primary">
          Start practice <ArrowRight className="h-5 w-5" aria-hidden="true" />
        </Link>
        <div className="home-quick-links">
          {destinations.slice(1).map(({ href, title }) => (
            <Link href={href} key={href}>
              {title} <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </Link>
          ))}
        </div>
      </section>
      <section className="home-range" aria-label="Button opening range preview">
        <div className="home-range-title">
          <span>BTN open</span>
          <span className="text-muted">
            6-max · {scenario?.effectiveStackBb ?? 100} bb
          </span>
        </div>
        <HandMatrix mode="display" strategy={strategy} squareCells />
        <div className="home-range-caption">
          <span className="inline-flex items-center gap-2">
            <i className="h-2 w-2 bg-raise" aria-hidden="true" />
            Raise
          </span>
          <Link href="/preflop">
            Explore ranges <ArrowRight className="h-4 w-4" aria-hidden="true" />
          </Link>
        </div>
      </section>
      <nav className="home-destinations" aria-label="Study tools">
        {destinations.map(({ href, title, detail, icon: Icon }) => (
          <Link href={href} key={href}>
            <Icon className="h-5 w-5 text-accent" aria-hidden="true" />
            <div>
              <h2>{title}</h2>
              <p>{detail}</p>
            </div>
            <ArrowRight className="h-5 w-5" aria-hidden="true" />
          </Link>
        ))}
      </nav>
    </div>
  );
}
