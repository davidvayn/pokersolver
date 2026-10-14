'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { parseBoard, type Card } from '@/lib/cards';
import { ACTIVE_FULL_HAND_MANIFESTS } from '@/lib/practice-models';
import { buildOpponentModel } from '@/lib/opponent-model';
import { PracticePolicyClient, type PinnedPracticeModel } from '@/lib/practice-policy-client';
import type { LegalAction } from '@/lib/practice-types';
import type { NodeStrategy } from '@/lib/solver/client';
import { replayStudyLine, studyDecisionKey, type StudyChoice } from '@/lib/solver/study-engine';

const profile = buildOpponentModel([], 'baseline');
export function useSolverStudy() {
  const [depthBb, setDepthBb] = useState(20);
  const [board, setBoard] = useState<Card[]>(() => parseBoard('Qh7s2c'));
  const [choices, setChoices] = useState<StudyChoice[]>([]);
  const [retry, setRetry] = useState(0);
  const [, setRevision] = useState(0);
  const [status, setStatus] = useState<{ key: string; pending: boolean; error: string | null }>({ key: '', pending: true, error: null });
  const client = useRef<PracticePolicyClient | null>(null);
  const cache = useRef(new Map<string, NodeStrategy>());
  if (!client.current) client.current = new PracticePolicyClient(undefined, undefined, ACTIVE_FULL_HAND_MANIFESTS);
  const manifest = ACTIVE_FULL_HAND_MANIFESTS.find((m) => m.depthsBb.includes(depthBb) && m.runtime?.kind === 'rust-continual-resolver-v1');
  const config = useMemo(() => manifest?.runtime?.kind === 'rust-continual-resolver-v1'
    ? { modelVersion: manifest.version, depthBb, actionAbstraction: manifest.runtime.actionAbstraction } : null, [manifest, depthBb]);
  const line = useMemo(() => config ? replayStudyLine(config, board, choices) : null, [config, board, choices]);
  const lastCompleteLine = useRef(line);
  useEffect(() => { if (line && !line.unappliedChoices) lastCompleteLine.current = line; }, [line]);
  const decisions = line?.unappliedChoices && lastCompleteLine.current
    ? [...line.decisions, ...lastCompleteLine.current.decisions.slice(line.decisions.length)] : line?.decisions ?? [];
  const target = line?.decisions.at(-1)?.state;
  const key = target ? studyDecisionKey(target) : '';
  const node = key ? cache.current.get(key) : undefined;
  useEffect(() => {
    if (!manifest || !target || !key || line?.unappliedChoices) return;
    if (cache.current.has(key)) { setStatus({ key, pending: false, error: null }); return; }
    let canceled = false;
    setStatus({ key, pending: true, error: null });
    const pinned: PinnedPracticeModel = { manifest, depthBb };
    client.current!.lookupState({ pinned, state: target, profile, usage: 'grading', includeRangeStrategy: true })
      .then((result) => {
        if (!result.rangeStrategy) throw new Error('The shared resolver did not return its public range');
        cache.current.set(key, result.rangeStrategy);
        if (cache.current.size > 64) cache.current.delete(cache.current.keys().next().value!);
        if (!canceled) { setRevision((value) => value + 1); setStatus({ key, pending: false, error: null }); }
      }).catch((error: unknown) => {
        if (!canceled) setStatus({ key, pending: false, error: error instanceof Error ? error.message : 'The shared resolver is unavailable' });
      });
    return () => { canceled = true; };
  }, [key, target, manifest, depthBb, retry, line?.unappliedChoices]);
  const error = !manifest ? 'No installed full-hand policy at this depth' : status.key === key ? status.error : null;
  const pending = Boolean(line?.unappliedChoices || (!node && !error) || (status.key === key && status.pending));
  function choose(index: number, action: LegalAction) {
    const decision = line?.decisions[index];
    if (!decision?.state.toAct || !decision.actions.some((legal) => legal.id === action.id)) return;
    setChoices((current) => [...current.slice(0, index), { street: decision.state.street, actor: decision.state.toAct!, action }]);
  }
  function changeDepth(depth: number) { setDepthBb(depth); setChoices([]); }
  return { manifest, depthBb, changeDepth, board, setBoard, choices, line, decisions, target, key, node,
    pending, error, choose, retry: () => setRetry((value) => value + 1),
    reset: () => setChoices([]), back: () => setChoices((current) => current.slice(0, -1)) };
}
