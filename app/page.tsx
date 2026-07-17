import Link from 'next/link';

const CARDS = [
  {
    href: '/calculator',
    title: 'Equity Calculator',
    desc: 'Hand vs hand, hand vs range, range vs range. Exact enumeration + Monte Carlo, computed off-thread.',
    icon: '％',
  },
  {
    href: '/charts',
    title: 'Preflop GTO Charts',
    desc: 'Position-aware opening and response ranges on a 13×13 matrix. Switch seats instantly.',
    icon: '♦',
  },
  {
    href: '/solver',
    title: 'Postflop Solver',
    desc: 'Discounted-CFR postflop solves that run in your browser via WebAssembly. No data leaves your machine.',
    icon: '♣',
  },
  {
    href: '/settings',
    title: 'AI Analysis',
    desc: 'Bring your own API key (Anthropic, OpenAI, …). Get natural-language reads on any spot.',
    icon: '✦',
  },
];

export default function Home() {
  return (
    <div className="flex flex-col gap-10 py-6">
      <section className="flex flex-col items-center gap-4 text-center">
        <div
          className="grid h-14 w-14 place-items-center rounded-2xl text-2xl text-accent-fg"
          style={{ background: 'rgb(var(--felt))' }}
        >
          ♠
        </div>
        <h1 className="max-w-2xl text-3xl font-bold tracking-tight sm:text-4xl">
          A local Texas Hold&apos;em solver
        </h1>
        <p className="max-w-xl text-muted">
          Accurate equity, GTO preflop charts, and real postflop solving — all
          computed on your machine. Add an API key for AI-powered spot analysis.
          Runs locally or deploys to Vercel.
        </p>
        <Link
          href="/calculator"
          className="rounded-md bg-accent px-5 py-2.5 text-sm font-semibold text-accent-fg hover:opacity-90"
        >
          Open the calculator
        </Link>
      </section>

      <section className="grid gap-4 sm:grid-cols-2">
        {CARDS.map((c) => (
          <Link
            key={c.href}
            href={c.href}
            className="group rounded-xl border border-border bg-surface p-5 transition-colors hover:border-accent"
          >
            <div className="mb-3 grid h-10 w-10 place-items-center rounded-lg bg-surface-2 text-lg">
              {c.icon}
            </div>
            <h2 className="mb-1 font-semibold group-hover:text-accent">
              {c.title}
            </h2>
            <p className="text-sm text-muted">{c.desc}</p>
          </Link>
        ))}
      </section>
    </div>
  );
}
