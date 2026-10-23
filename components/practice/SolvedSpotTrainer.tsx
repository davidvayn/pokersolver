'use client';

import { useEffect, useRef, useState } from 'react';
import { PracticeTable, type TableStatus } from '@/components/practice/PracticeTable';
import { advanceSpot, loadSolvedSpot, playSpotOpponent, spotActions, spotEvLoss, startSpot,
  type SolvedSpot, type SpotCatalogEntry, type SpotSession } from '@/lib/solved-spots';
import type { LegalAction, PolicyAction, Seat } from '@/lib/practice-types';

interface DecisionReview { actions: PolicyAction[]; chosen: string; loss: number | null }
const buttonClass = 'min-h-11 rounded-md border border-border bg-surface px-3 py-2 text-sm font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent';

export function SolvedSpotTrainer({ entries }: { entries: SpotCatalogEntry[] }) {
  const [entryId, setEntryId] = useState(entries[0]?.id ?? '');
  const entry = entries.find((item) => item.id === entryId);
  const [spot, setSpot] = useState<SolvedSpot | null>(null);
  const [session, setSession] = useState<SpotSession | null>(null);
  const [hero, setHero] = useState<Seat>('big-blind');
  const [status, setStatus] = useState<TableStatus>('loading');
  const [reviews, setReviews] = useState<DecisionReview[]>([]);
  const [error, setError] = useState('');
  const [retry, setRetry] = useState(0);
  const feedbackRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const abort = new AbortController();
    setStatus('loading'); setError(''); setSpot(null); setSession(null); setReviews([]);
    if (!entry) { setError('No verified spots are installed.'); setStatus('error'); return; }
    loadSolvedSpot(entry, abort.signal).then((loaded) => {
      if (abort.signal.aborted) return;
      setSpot(loaded);
      // Seat changes apply to the next deal, not to a hand in progress.
      setSession(playSpotOpponent(loaded, startSpot(loaded, entry.sha256, 'big-blind')));
      setStatus('decision');
    }).catch((cause: unknown) => {
      if (abort.signal.aborted) return;
      setError(cause instanceof Error ? cause.message : 'Unable to load saved spot.'); setStatus('error');
    });
    return () => abort.abort();
  }, [entry, retry]);

  function fail(cause: unknown) {
    setError(cause instanceof Error ? cause.message : 'The saved solution could not be read.'); setStatus('error');
  }
  function nextDeal() {
    if (!spot || !entry) return;
    try {
      const next = playSpotOpponent(spot, startSpot(spot, entry.sha256, hero));
      setSession(next); setReviews([]); setStatus(next.state.terminal ? 'review' : 'decision');
    } catch (cause) { fail(cause); }
  }
  function choose(action: LegalAction) {
    if (!spot || !session || status !== 'decision' || session.state.toAct !== session.state.hero) return;
    try {
      const actions = spotActions(spot, session);
      const next = advanceSpot(spot, session, action.id);
      setReviews((previous) => [...previous, { actions, chosen: action.id, loss: spotEvLoss(actions, action.id) }]);
      setSession(next); setStatus('feedback');
    } catch (cause) { fail(cause); }
  }
  function continueHand() {
    if (status === 'review') { nextDeal(); return; }
    if (!spot || !session || status !== 'feedback') return;
    try {
      const next = playSpotOpponent(spot, session);
      setSession(next); setStatus(next.state.terminal ? 'review' : 'decision');
    } catch (cause) { fail(cause); }
  }

  let actions: PolicyAction[] = [];
  let lookupError = '';
  if (spot && session && status === 'decision') {
    try { actions = spotActions(spot, session); } catch (cause) { lookupError = cause instanceof Error ? cause.message : 'Missing saved strategy'; }
  }
  const lastReview = reviews.at(-1);
  const showReview = lastReview && (status === 'feedback' || status === 'review');
  return (
    <>
      {entries.length === 1 && <p className="mb-3 text-sm font-semibold">{entry?.title}</p>}
      <div className="mb-5 flex flex-wrap items-end gap-4">
        {entries.length > 1 && <label className="grid gap-1 text-sm font-medium">Saved spot
          <select className={buttonClass} value={entryId} onChange={(event) => setEntryId(event.target.value)}>
            {entries.map((item) => <option key={item.id} value={item.id}>{item.title}</option>)}
          </select>
        </label>}
        <label className="grid gap-1 text-sm font-medium">Your seat · next deal
          <select className={buttonClass} value={hero} onChange={(event) => setHero(event.target.value as Seat)}>
            <option value="big-blind">Big blind · first to act</option>
            <option value="button-small-blind">Button · in position</option>
          </select>
        </label>
        <button className={buttonClass} disabled={!spot || status === 'loading'} onClick={nextDeal}>New deal</button>
        {session && session.state.hero !== hero && <p className="text-sm text-muted" role="status">Seat change applies on the next deal.</p>}
      </div>
      <div className="grid min-w-0 gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
        <PracticeTable state={session?.state ?? null} node={actions.length ? { actions } : null}
          status={lookupError ? 'error' : status} mode="postflop" allModeledActions initialEventLabel="River spot loaded" showTableHistory={false}
          errorMessage={lookupError || error} revealOpponent={Boolean(session?.state.terminal && status === 'review')}
          selectedActionId={status === 'feedback' ? lastReview?.chosen ?? null : null}
          onAction={choose} onContinue={continueHand} onRetry={() => setRetry((v) => v+1)}
          onOpenAnalyst={() => feedbackRef.current?.focus()} />
        <aside className="min-w-0 space-y-6" aria-label="Saved solution and feedback">
          <section ref={feedbackRef} tabIndex={-1} aria-labelledby="spot-feedback-title" className="scroll-mt-4 rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent">
            <h2 id="spot-feedback-title" className="text-lg font-semibold">Decision feedback</h2>
            {showReview ? <div className="mt-3" role="status" aria-live="polite">
              <p className="text-sm">Decision {reviews.length} · {lastReview.actions.find((a) => a.id === lastReview.chosen)?.label}</p>
              <p className="mt-2 text-xl font-semibold">{lastReview.loss === null ? 'EV unavailable at this off-path node' : `${lastReview.loss.toFixed(3)}bb EV loss`}</p>
              <div className="mt-3 overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <caption className="sr-only">Saved strategy frequencies and action EVs for your last decision</caption>
                  <thead><tr className="border-b border-border"><th className="py-2 pr-2">Action</th><th className="p-2 text-right">Mix</th><th className="py-2 pl-2 text-right">EV (bb)</th></tr></thead>
                  <tbody>{lastReview.actions.map((a) => <tr key={a.id} className={`border-b border-border ${a.id === lastReview.chosen ? 'font-semibold text-accent' : ''}`}>
                    <th scope="row" className="py-3 pr-2 font-medium">{a.label}{a.id === lastReview.chosen ? ' · chosen' : ''}</th>
                    <td className="p-2 text-right font-mono">{(a.probability*100).toFixed(1)}%</td>
                    <td className="py-2 pl-2 text-right font-mono">{a.evBb?.toFixed(3) ?? '—'}</td>
                  </tr>)}</tbody>
                </table>
              </div>
              <p className="mt-3 text-xs leading-5 text-muted">EVs enumerate the saved strategy through showdown. They are net from the hand start, including earlier commitments. Sampling SE is zero; numerical tolerance and residual strategy error are not.</p>
            </div> : <p className="mt-3 text-sm leading-6 text-muted">Choose an action at the table. The full mix and action EVs appear after your decision.</p>}
            {session?.state.terminal && status === 'review' && <p className="mt-3 text-sm" role="status">
              {session.state.result?.reason === 'fold' ? 'Hand ended by fold.' : 'Showdown.'} Your net result: {session.state.result?.netBb[session.state.hero].toFixed(1)}bb.
            </p>}
          </section>
          {session && session.state.actionHistory.length > 0 && <section aria-labelledby="spot-history-title" className="border-t border-border pt-5">
            <h2 id="spot-history-title" className="text-lg font-semibold">River action history</h2>
            <ol className="mt-3 space-y-2 text-sm" aria-label="Complete river action history">
              {session.state.actionHistory.map((action) => <li key={action.id} className="flex flex-wrap justify-between gap-2 border-b border-border py-2">
                <span className="text-muted">{action.actor === session.state.hero ? 'You' : 'Opponent'}</span><strong>{action.label}</strong>
              </li>)}
            </ol>
          </section>}
          {spot && <section aria-labelledby="spot-scope-title" className="border-t border-border pt-5">
            <h2 id="spot-scope-title" className="text-lg font-semibold">What is solved?</h2>
            <p className="mt-2 text-sm leading-6">{spot.input.description}</p>
            <dl className="mt-3 space-y-2 text-sm">
              <div><dt className="text-muted">Total best-response gain · this spot only</dt><dd className="font-mono">{spot.verification.totalBestResponseGainBb.toFixed(5)}bb / ≤0.05000bb</dd></div>
              <div><dt className="text-muted">Complete saved decision nodes</dt><dd>{spot.nodes.length}</dd></div>
              <div><dt className="text-muted">Compatible private-card deals checked</dt><dd>{spot.verification.compatibleDeals.toLocaleString()}</dd></div>
            </dl>
            <details className="mt-4 text-sm">
              <summary className="cursor-pointer py-2 font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent">Ranges, accuracy and version</summary>
              <p className="mt-2 leading-6 text-muted">{spot.input.rangeAssumptions}</p>
              <p className="mt-3 break-words"><strong>BTN:</strong> {spot.input.rangeClasses[0].join(', ')}</p>
              <p className="mt-2 break-words"><strong>BB:</strong> {spot.input.rangeClasses[1].join(', ')}</p>
              <p className="mt-3 leading-6 text-muted">Bets: {spot.input.betPotFractions.map((f) => `${f*100}% pot`).join(', ')}, plus all-in. Raises: {spot.input.raisePotFractions.map((f) => `${f*100}% of the pot after calling`).join(', ')}, plus all-in. Raise cap: {spot.input.raiseCap} after the initial bet.</p>
              <p className="mt-3 leading-6 text-muted">Direct pair enumeration checks every saved action EV to 0.0001bb and sums both players’ best-response gains. This is a conditional river solution, not a full-hand exploitability certificate. No online resolver or neural values are used.</p>
              <p className="mt-3 break-all font-mono text-xs">Version SHA-256: {entry?.sha256}</p>
            </details>
          </section>}
        </aside>
      </div>
    </>
  );
}
