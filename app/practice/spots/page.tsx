import Link from 'next/link';
import { PageHeading } from '@/components/design/DesignShell';
import { SolvedSpotTrainer } from '@/components/practice/SolvedSpotTrainer';
import catalog from '@/data/solved-spots/catalog.json';
import type { SpotCatalogEntry } from '@/lib/solved-spots';

export default function SolvedSpotsPage() {
  return (
    <div className="practice-study pb-8">
      <PageHeading title="Solved spots" meta={<span className="hidden sm:inline">Complete saved strategies · no online solve</span>}>
        <Link href="/practice" className="study-button-secondary">Full hands</Link>
      </PageHeading>
      <p className="mb-6 max-w-3xl text-sm leading-6 text-muted">
        Verified river strategies for the stated ranges and betting sizes—not full-hand GTO.
      </p>
      <SolvedSpotTrainer entries={catalog as SpotCatalogEntry[]} />
    </div>
  );
}
