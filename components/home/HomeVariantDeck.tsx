import Link from 'next/link';
import { ArrowRight, Layers3, Target } from 'lucide-react';

function PlayingCard({
  rank,
  suit,
  className = '',
}: {
  rank: string;
  suit: '♠' | '♥' | '♦' | '♣';
  className?: string;
}) {
  const red = suit === '♥' || suit === '♦';

  return (
    <div
      className={`flex aspect-[5/7] w-24 flex-col justify-between rounded-lg border border-black/15 bg-white p-3 font-mono font-bold shadow-xl sm:w-28 ${className}`}
      aria-hidden="true"
    >
      <span className={red ? 'text-raise' : 'text-black'}>{rank}</span>
      <span
        className={
          'self-center text-4xl ' + (red ? 'text-raise' : 'text-black')
        }
        aria-hidden="true"
      >
        {suit}
      </span>
      <span
        className={
          'rotate-180 self-end ' + (red ? 'text-raise' : 'text-black')
        }
        aria-hidden="true"
      >
        {rank}
      </span>
    </div>
  );
}

export function HomeVariantDeck() {
  return (
    <div className="-mb-6 -mt-6">
      <section
        aria-labelledby="deck-title"
        className="relative left-1/2 min-h-[calc(100svh-3.5rem)] w-screen -translate-x-1/2 overflow-hidden bg-bg text-fg"
      >
        <div
          aria-hidden="true"
          className="absolute inset-y-0 left-[18%] hidden w-px bg-fg/15 lg:block"
        />
        <div
          aria-hidden="true"
          className="absolute inset-y-0 right-[18%] hidden w-px bg-fg/15 lg:block"
        />

        <div className="relative mx-auto flex min-h-[calc(100svh-3.5rem)] w-full max-w-[1600px] flex-col px-4 pb-24 pt-7 sm:px-8 sm:pb-14 lg:px-12">
          <header className="flex items-center justify-between border-b border-border pb-4">
            <p className="flex items-center gap-2 font-mono text-xs font-semibold text-accent">
              <Layers3 className="h-4 w-4" aria-hidden="true" />
              THE STUDY DECK
            </p>
            <p className="font-mono text-xs text-muted">DEAL / DECIDE / REVIEW</p>
          </header>

          <div className="relative flex flex-1 items-start pb-8 pt-20 sm:items-center sm:py-10">
            <div
              role="img"
              aria-label="Ace-king suited being dealt onto a full-width study table"
              className="pointer-events-none absolute inset-0 [perspective:1200px]"
            >
              <div className="absolute inset-x-[3%] bottom-[2%] top-[70%] border border-fg/30 [transform:rotateX(58deg)_rotateZ(-4deg)] sm:inset-x-[10%] sm:bottom-[4%] sm:top-[76%] sm:[transform:rotateX(58deg)_rotateZ(-7deg)]">
                <div className="absolute inset-[8%] border border-accent/60" />
                <p className="absolute bottom-4 left-5 hidden font-mono text-xs text-muted sm:block">
                  TABLE 01 / BUTTON
                </p>
              </div>

              <div className="absolute bottom-[8%] left-[24%] [transform:rotateY(8deg)] sm:bottom-auto sm:left-[62%] sm:top-[18%]">
                <PlayingCard
                  rank="A"
                  suit="♣"
                  className="home-deal-left !w-20 sm:!w-36"
                />
              </div>
              <div className="absolute bottom-[7%] right-[23%] [transform:rotateY(-8deg)] sm:bottom-auto sm:right-[14%] sm:top-[28%]">
                <PlayingCard
                  rank="K"
                  suit="♣"
                  className="home-deal-right !w-20 sm:!w-36"
                />
              </div>

              <div className="absolute bottom-[14%] right-[7%] hidden border border-border bg-surface px-4 py-3 shadow-[10px_10px_0_rgb(var(--accent))] sm:block">
                <p className="font-mono text-[10px] text-muted">SPOT 024</p>
                <p className="mt-1 text-sm font-semibold text-fg">
                  BTN open / A♣K♣
                </p>
              </div>
            </div>

            <div className="relative z-10 w-full max-w-4xl">
              <p className="font-mono text-xs font-semibold text-accent">
                TRAIN THE DECISION BEFORE IT COUNTS
              </p>
              <h1
                id="deck-title"
                className="mt-5 max-w-5xl text-6xl font-bold leading-[0.9] text-fg sm:text-8xl lg:text-[9rem]"
              >
                Poker Lab
              </h1>
              <p className="mt-7 max-w-2xl text-lg leading-8 text-muted sm:text-xl">
                Every hand is a prompt. Deal yourself focused spots, choose the
                action, and turn feedback into a repeatable edge.
              </p>
              <div className="mt-8 flex gap-2 sm:gap-3">
                <Link
                  href="/practice"
                  className="inline-flex min-h-12 flex-1 items-center justify-center gap-2 bg-accent px-3 py-3 font-semibold text-accent-fg transition-opacity hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg sm:flex-none sm:px-6"
                >
                  <Target className="h-5 w-5" aria-hidden="true" />
                  Start practice
                  <ArrowRight
                    className="hidden h-4 w-4 sm:block"
                    aria-hidden="true"
                  />
                </Link>
                <Link
                  href="/preflop"
                  className="inline-flex min-h-12 flex-1 items-center justify-center border border-border bg-surface/70 px-3 py-3 font-semibold text-fg transition-colors hover:border-accent hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg sm:flex-none sm:px-6"
                >
                  Browse ranges
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
