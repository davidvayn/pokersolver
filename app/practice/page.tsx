'use client';

import Link from 'next/link';
import { useCallback, useEffect, useRef, useState } from 'react';
import { BarChart3, Info, Settings2, Target, X } from 'lucide-react';
import {
  AnalystRail,
  type RailTab,
} from '@/components/practice/AnalystRail';
import { PracticeTable } from '@/components/practice/PracticeTable';
import {
  applyAction,
  canonicalPolicyHash,
  createHand,
  handBucket,
  isPreflopRoundComplete,
  otherSeat,
  stopForReview,
} from '@/lib/practice-engine';
import {
  gradePolicyChoice,
  samplePolicyAction,
  validatePolicyNode,
} from '@/lib/practice-grading';
import {
  loadPracticeSettings,
  nextHeroSeat,
  savePracticeSettings,
  structuralSettingsChanged,
} from '@/lib/practice';
import {
  loadPracticeHands,
  savePracticeHand,
  subscribePracticeHistory,
} from '@/lib/practice-history';
import {
  adaptationConfigForRuntime,
  buildOpponentModel,
} from '@/lib/opponent-model';
import { PUSH_FOLD_MANIFEST } from '@/lib/practice-models';
import {
  PolicyUnavailableError,
  PracticePolicyClient,
  type PinnedPracticeModel,
} from '@/lib/practice-policy-client';
import {
  createPushFoldSpot,
  finishPushFoldHand,
  type PushFoldSpot,
} from '@/lib/push-fold-policy';
import {
  DEFAULT_PRACTICE_SETTINGS,
  type HandState,
  type LegalAction,
  type OpponentModelSnapshot,
  type OpponentPolicyTrace,
  type PolicyManifest,
  type PolicyNode,
  type PracticeDecisionRecord,
  type PracticeHandRecord,
  type PracticeSettings,
} from '@/lib/practice-types';

type TableStatus =
  | 'loading'
  | 'unavailable'
  | 'decision'
  | 'feedback'
  | 'review'
  | 'error';

function unavailableCopy(settings: PracticeSettings): string {
  if (settings.mode === 'postflop') {
    return 'No accepted postflop sample and policy shards are installed. The table will not replay or score a synthetic line.';
  }
  if (settings.mode === 'preflop') {
    return 'No accepted full no-limit model is installed for a complete preflop round. Push/fold remains available as its validated subtype.';
  }
  return `The ${settings.depthBb}bb full-hand seeds have not passed every activation gate. The depth remains hidden and the table will not substitute a strategy.`;
}

function fullHandDepths(manifests: PolicyManifest[]): number[] {
  return [
    ...new Set(
      manifests
        .filter(
          (manifest) =>
            manifest.subtype === 'full-hand' &&
            manifest.active &&
            manifest.validation.status === 'accepted'
        )
        .flatMap((manifest) => manifest.depthsBb)
        .filter((depth) => [20, 50, 100].includes(depth))
    ),
  ].sort((first, second) => first - second);
}

function stopAfterPreflop(state: HandState): HandState {
  return stopForReview(
    { ...state, street: 'preflop', board: [] },
    'preflop-complete'
  );
}

async function checkedPolicyNode(
  client: PracticePolicyClient,
  pinned: PinnedPracticeModel,
  state: HandState,
  profile: OpponentModelSnapshot,
  usage: 'grading' | 'opponent'
): Promise<{ node: PolicyNode; trace: OpponentPolicyTrace | null }> {
  const result = await client.lookupState({ pinned, state, profile, usage });
  const { node } = result;
  const expectedHash = await canonicalPolicyHash(state);
  const errors = validatePolicyNode(node);
  if (node.stateHash !== expectedHash) errors.push('Policy node hash mismatch');
  for (const action of node.actions) {
    try {
      applyAction(state, action);
    } catch (error) {
      errors.push(
        `${action.id} is illegal: ${error instanceof Error ? error.message : 'unknown error'}`
      );
    }
  }
  if (errors.length > 0) {
    throw new PolicyUnavailableError(
      `The pinned policy node failed validation: ${errors.join('; ')}`
    );
  }
  return result;
}

async function advancePolicyToHero(input: {
  client: PracticePolicyClient;
  pinned: PinnedPracticeModel;
  state: HandState;
  mode: PracticeSettings['mode'];
  profile: OpponentModelSnapshot;
  onProgress: (state: HandState) => void;
  onOpponentPolicy: (trace: OpponentPolicyTrace) => void;
}): Promise<{ state: HandState; node: PolicyNode | null }> {
  let state = input.state;
  while (!state.terminal && state.toAct !== state.hero) {
    const lookup = await checkedPolicyNode(
      input.client,
      input.pinned,
      state,
      input.profile,
      'opponent'
    );
    if (lookup.trace) input.onOpponentPolicy(lookup.trace);
    const previous = state;
    state = applyAction(state, samplePolicyAction(lookup.node.actions));
    input.onProgress(state);
    if (
      input.mode === 'preflop' &&
      isPreflopRoundComplete(previous, state)
    ) {
      state = stopAfterPreflop(state);
      input.onProgress(state);
      return { state, node: null };
    }
  }
  if (state.terminal) return { state, node: null };
  return {
    state,
    node: (
      await checkedPolicyNode(
        input.client,
        input.pinned,
        state,
        input.profile,
        'grading'
      )
    ).node,
  };
}

export default function PracticePage() {
  const [settings, setSettings] = useState<PracticeSettings>(
    DEFAULT_PRACTICE_SETTINGS
  );
  const [pendingSettings, setPendingSettings] =
    useState<PracticeSettings | null>(null);
  const [spot, setSpot] = useState<PushFoldSpot | null>(null);
  const [state, setState] = useState<HandState | null>(null);
  const [activeNode, setActiveNode] = useState<PolicyNode | null>(null);
  const [manifests, setManifests] = useState<PolicyManifest[]>([
    PUSH_FOLD_MANIFEST,
  ]);
  const [handManifest, setHandManifest] =
    useState<PolicyManifest | null>(null);
  const [currentHandDecisions, setCurrentHandDecisions] = useState<
    PracticeDecisionRecord[]
  >([]);
  const [status, setStatus] = useState<TableStatus>('unavailable');
  const [errorMessage, setErrorMessage] = useState('');
  const [selectedActionId, setSelectedActionId] = useState<string | null>(null);
  const [feedback, setFeedback] =
    useState<PracticeDecisionRecord | null>(null);
  const [recentHands, setRecentHands] = useState<PracticeHandRecord[]>([]);
  const [opponentModel, setOpponentModel] =
    useState<OpponentModelSnapshot | null>(null);
  const [sessionDecisions, setSessionDecisions] = useState<
    PracticeDecisionRecord[]
  >([]);
  const [completedHands, setCompletedHands] = useState(0);
  const [railTab, setRailTab] = useState<RailTab>('settings');
  const [historyWarning, setHistoryWarning] = useState('');
  const [goalSummary, setGoalSummary] = useState(false);
  const [mobileRailOpen, setMobileRailOpen] = useState(false);
  const decisionStartedAt = useRef(0);
  const handStartedAt = useRef(0);
  const requestId = useRef(0);
  const policyClientRef = useRef<PracticePolicyClient | null>(null);
  const pinnedModelRef = useRef<PinnedPracticeModel | null>(null);
  const recentHandsRef = useRef<PracticeHandRecord[]>([]);
  const opponentModelRef = useRef<OpponentModelSnapshot | null>(null);
  const opponentQueriesRef = useRef<OpponentPolicyTrace[]>([]);
  const goalTargetRef = useRef<number | null>(null);
  const goalReachedRef = useRef(false);
  const mobileSheetRef = useRef<HTMLElement | null>(null);
  const mobileCloseRef = useRef<HTMLButtonElement | null>(null);
  if (!policyClientRef.current) {
    policyClientRef.current = new PracticePolicyClient();
  }

  const beginHand = useCallback(
    async (nextSettings: PracticeSettings, handNumber: number) => {
      const currentRequest = ++requestId.current;
      setSelectedActionId(null);
      setFeedback(null);
      setErrorMessage('');
      setGoalSummary(false);
      setActiveNode(null);
      setCurrentHandDecisions([]);
      setHandManifest(null);
      setOpponentModel(null);
      opponentModelRef.current = null;
      opponentQueriesRef.current = [];
      goalReachedRef.current = false;
      pinnedModelRef.current = null;

      if (nextSettings.mode === 'postflop') {
        setSpot(null);
        setState(null);
        setHandManifest(null);
        pinnedModelRef.current = null;
        setStatus('unavailable');
        setRailTab('settings');
        return;
      }

      setStatus('loading');
      try {
        if (nextSettings.mode === 'push-fold') {
          const nextSpot = await createPushFoldSpot({
            depthBb: nextSettings.pushFoldDepthBb,
            hero: nextHeroSeat(nextSettings.heroSeat, handNumber),
            handNumber,
          });
          if (currentRequest !== requestId.current) return;
          pinnedModelRef.current = null;
          setHandManifest(PUSH_FOLD_MANIFEST);
          setSpot(nextSpot);
          setState(nextSpot.state);
          setActiveNode(nextSpot.node);
          setStatus('decision');
          handStartedAt.current = Date.now();
          decisionStartedAt.current = performance.now();
          return;
        }

        const client = policyClientRef.current as PracticePolicyClient;
        const pinned = await client.pinFullHandModel(nextSettings.depthBb);
        if (currentRequest !== requestId.current) return;
        pinnedModelRef.current = pinned;
        const neuralRuntime =
          pinned.manifest.runtime?.kind === 'neural-deep-cfr-v1'
            ? pinned.manifest.runtime
            : undefined;
        const profile = buildOpponentModel(
          recentHandsRef.current,
          neuralRuntime ? nextSettings.opponentStyle : 'baseline',
          adaptationConfigForRuntime(neuralRuntime)
        );
        opponentModelRef.current = profile;
        setOpponentModel(profile);
        setHandManifest(pinned.manifest);
        setSpot(null);
        handStartedAt.current = Date.now();

        // A BB drill is conditioned on the opponent taking an action that
        // actually reaches the hero. Terminal folds before the first hero
        // decision are authentic but not useful practice spots, so resample.
        for (let attempt = 0; attempt < 128; attempt++) {
          const initial = createHand({
            id: `full-${handNumber}-${attempt}-${pinned.manifest.version}`,
            modelVersion: pinned.manifest.version,
            depthBb: nextSettings.depthBb,
            button: 'button-small-blind',
            hero: nextHeroSeat(nextSettings.heroSeat, handNumber),
          });
          if (currentRequest !== requestId.current) return;
          setState(initial);
          const advanced = await advancePolicyToHero({
            client,
            pinned,
            state: initial,
            mode: nextSettings.mode,
            profile,
            onProgress: (progress) => {
              if (currentRequest === requestId.current) setState(progress);
            },
            onOpponentPolicy: (trace) => {
              if (currentRequest === requestId.current) {
                opponentQueriesRef.current.push(trace);
              }
            },
          });
          if (currentRequest !== requestId.current) return;
          if (advanced.state.terminal) continue;
          setState(advanced.state);
          setActiveNode(advanced.node);
          setStatus('decision');
          decisionStartedAt.current = performance.now();
          return;
        }
        throw new PolicyUnavailableError(
          'No reachable hero decision was sampled from the pinned model.'
        );
      } catch (error) {
        if (currentRequest !== requestId.current) return;
        setStatus(
          pinnedModelRef.current || !(error instanceof PolicyUnavailableError)
            ? 'error'
            : 'unavailable'
        );
        setErrorMessage(
          error instanceof Error ? error.message : 'Could not load the practice spot.'
        );
      }
    },
    []
  );

  useEffect(() => {
    const initialize = async () => {
      const loaded = loadPracticeSettings();
      let availableManifests = [PUSH_FOLD_MANIFEST];
      try {
        availableManifests =
          (await policyClientRef.current?.loadManifests()) ??
          availableManifests;
      } catch {
        // beginHand will show a fail-closed unavailable state below.
      }
      setManifests(availableManifests);
      const hands = await loadPracticeHands(500);
      recentHandsRef.current = hands;
      setRecentHands(hands.slice(0, 100));
      const availableDepths = fullHandDepths(availableManifests);
      const persistenceSafe =
        loaded.dealMode === 'adaptive'
          ? { ...loaded, dealMode: 'authentic' as const }
          : loaded;
      const effective =
        (persistenceSafe.mode === 'full-hand' || persistenceSafe.mode === 'preflop') &&
        availableDepths.length > 0 &&
        !availableDepths.includes(persistenceSafe.depthBb)
          ? {
              ...persistenceSafe,
              depthBb: availableDepths[0] as PracticeSettings['depthBb'],
            }
          : persistenceSafe;
      setSettings(effective);
      if (effective !== loaded) savePracticeSettings(effective);
      goalTargetRef.current =
        effective.decisionGoal === 'continuous'
          ? null
          : effective.decisionGoal;
      void beginHand(effective, 0);
    };
    void initialize();
  }, [beginHand]);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      const hands = await loadPracticeHands(500);
      if (active) {
        recentHandsRef.current = hands;
        setRecentHands(hands.slice(0, 100));
      }
    };
    const unsubscribe = subscribePracticeHistory(() => void refresh());
    return () => {
      active = false;
      unsubscribe();
    };
  }, []);

  useEffect(() => {
    if (!mobileRailOpen) return;
    const close = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setMobileRailOpen(false);
        return;
      }
      if (event.key !== 'Tab') return;
      const focusable = mobileSheetRef.current?.querySelectorAll<HTMLElement>(
        'button:not(:disabled), select:not(:disabled), summary, a[href], [tabindex]:not([tabindex="-1"])'
      );
      if (!focusable?.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener('keydown', close);
    mobileCloseRef.current?.focus();
    return () => window.removeEventListener('keydown', close);
  }, [mobileRailOpen]);

  function updateSettings(next: PracticeSettings) {
    savePracticeSettings(next);
    if (settings.decisionGoal !== next.decisionGoal) {
      goalTargetRef.current =
        next.decisionGoal === 'continuous'
          ? null
          : sessionDecisions.length + next.decisionGoal;
      goalReachedRef.current = false;
      setGoalSummary(false);
    }
    if (
      (status === 'decision' || status === 'feedback') &&
      state &&
      !state.terminal &&
      structuralSettingsChanged(settings, next)
    ) {
      setPendingSettings(next);
      return;
    }
    setSettings(next);
    setPendingSettings(null);
    if (structuralSettingsChanged(settings, next)) {
      void beginHand(next, completedHands);
    }
  }

  async function finishAndSave(
    finished: HandState,
    decisions: PracticeDecisionRecord[],
    decisionCount: number
  ) {
    if (!finished.result) throw new Error('Completed practice hand has no result');
    const completedAt = Date.now();
    const opponent = otherSeat(finished.hero);
    const handRecord: PracticeHandRecord = {
      id: finished.id,
      startedAt: handStartedAt.current || completedAt,
      completedAt,
      modelVersion: finished.modelVersion,
      mode: settings.mode,
      depthBb: finished.depthBb,
      button: finished.button,
      hero: finished.hero,
      heroCards: [...finished.holeCards[finished.hero]],
      opponentCards: [...finished.holeCards[opponent]],
      board: [...finished.board],
      actions: finished.actionHistory,
      decisions,
      opponentModel: opponentModelRef.current ?? undefined,
      opponentPolicyQueries: [...opponentQueriesRef.current],
      result: finished.result,
    };
    setState(finished);
    setActiveNode(null);
    const profileHistory = [
      handRecord,
      ...recentHandsRef.current.filter((hand) => hand.id !== handRecord.id),
    ].slice(0, 500);
    recentHandsRef.current = profileHistory;
    setRecentHands(profileHistory.slice(0, 100));
    setCompletedHands((current) => current + 1);
    setStatus('review');
    setRailTab('feedback');
    setMobileRailOpen(true);
    if (
      goalTargetRef.current !== null &&
      decisionCount >= goalTargetRef.current
    ) {
      goalReachedRef.current = true;
    }
    if (goalReachedRef.current) {
      setGoalSummary(true);
      setRailTab('stats');
    }
    const saved = await savePracticeHand(handRecord);
    if (!saved) {
      setHistoryWarning(
        'This hand remains in the current run but could not be saved to IndexedDB.'
      );
    }
  }

  async function chooseAction(action: LegalAction) {
    const node = spot?.node ?? activeNode;
    if (status !== 'decision' || !state || !node || selectedActionId) return;
    setSelectedActionId(action.id);
    const responseMs = Math.max(
      0,
      performance.now() - decisionStartedAt.current
    );
    try {
      const grade = gradePolicyChoice(node, action.id);
      const previous = state;
      let next =
        settings.mode === 'push-fold'
          ? finishPushFoldHand(state, action)
          : applyAction(state, action);
      if (
        settings.mode === 'preflop' &&
        isPreflopRoundComplete(previous, next) &&
        !next.terminal
      ) {
        next = stopAfterPreflop(next);
      }
      const answeredAt = Date.now();
      const latestOpponentAction = [...state.actionHistory]
        .reverse()
        .find(
          (historyAction) =>
            historyAction.street === state.street &&
            historyAction.actor !== state.hero
        );
      const decision: PracticeDecisionRecord = {
        id: `${state.id}-${state.actionHistory.length}-${answeredAt}`,
        handId: state.id,
        answeredAt,
        responseMs,
        modelVersion: state.modelVersion,
        mode: settings.mode,
        depthBb: state.depthBb,
        street: state.street,
        position: state.hero,
        handBucket: handBucket(state.holeCards[state.hero]),
        facingAction:
          settings.mode === 'push-fold'
            ? state.hero === 'button-small-blind'
              ? 'first in'
              : 'BTN / SB all-in'
            : latestOpponentAction?.label ??
              (state.street === 'preflop' ? 'first in' : 'checked to'),
        stateHash: node.stateHash,
        board: [...next.board],
        heroCards: [...state.holeCards[state.hero]],
        chosenAction: action,
        policyActions: node.actions,
        ...grade,
        opponentModel: opponentModelRef.current ?? undefined,
      };
      const handDecisions = [...currentHandDecisions, decision];
      const decisionCount = sessionDecisions.length + 1;
      setState(next);
      setFeedback(decision);
      setCurrentHandDecisions(handDecisions);
      setSessionDecisions((current) => [...current, decision]);
      setRailTab('feedback');
      setMobileRailOpen(true);
      if (
        goalTargetRef.current !== null &&
        decisionCount >= goalTargetRef.current
      ) {
        goalReachedRef.current = true;
      }
      if (next.terminal) {
        await finishAndSave(next, handDecisions, decisionCount);
      } else {
        setStatus('feedback');
      }
    } catch (error) {
      setSelectedActionId(null);
      setStatus('error');
      setErrorMessage(
        error instanceof Error ? error.message : 'The action could not be applied.'
      );
    }
  }

  async function resumeFullHand() {
    const pinned = pinnedModelRef.current;
    const profile = opponentModelRef.current;
    if (!state || !pinned || !profile || state.terminal) {
      void beginHand(settings, completedHands);
      return;
    }
    const currentRequest = ++requestId.current;
    setStatus('loading');
    setSelectedActionId(null);
    setErrorMessage('');
    try {
      const advanced = await advancePolicyToHero({
        client: policyClientRef.current as PracticePolicyClient,
        pinned,
        state,
        mode: settings.mode,
        profile,
        onProgress: (progress) => {
          if (currentRequest === requestId.current) setState(progress);
        },
        onOpponentPolicy: (trace) => {
          if (currentRequest === requestId.current) {
            opponentQueriesRef.current.push(trace);
          }
        },
      });
      if (currentRequest !== requestId.current) return;
      setState(advanced.state);
      if (advanced.state.terminal) {
        await finishAndSave(
          advanced.state,
          currentHandDecisions,
          sessionDecisions.length
        );
        return;
      }
      setActiveNode(advanced.node);
      setStatus('decision');
      decisionStartedAt.current = performance.now();
    } catch (error) {
      if (currentRequest !== requestId.current) return;
      setStatus('error');
      setErrorMessage(
        error instanceof Error
          ? error.message
          : 'The pinned continuation policy is unavailable.'
      );
    }
  }

  function continueHand() {
    if (status === 'feedback') {
      void resumeFullHand();
      return;
    }
    const next = pendingSettings ?? settings;
    if (goalSummary && next.decisionGoal !== 'continuous') {
      goalTargetRef.current =
        (goalTargetRef.current ?? sessionDecisions.length) +
        next.decisionGoal;
      goalReachedRef.current = false;
    }
    if (pendingSettings) {
      setSettings(pendingSettings);
      savePracticeSettings(pendingSettings);
      setPendingSettings(null);
    }
    void beginHand(next, completedHands);
  }

  function retry() {
    if (
      status === 'error' &&
      state &&
      !state.terminal &&
      pinnedModelRef.current
    ) {
      void resumeFullHand();
      return;
    }
    void beginHand(settings, completedHands);
  }

  const depths = fullHandDepths(manifests);
  const manifest = handManifest;
  const modelLabel = manifest
    ? `${manifest.label} · ${manifest.version}`
    : 'No active model';

  return (
    <div className="pb-8">
      <header className="mb-5 flex flex-wrap items-start justify-between gap-4 border-b border-border pb-5">
        <div>
          <div className="flex items-center gap-2 font-mono text-xs font-semibold uppercase text-accent">
            <Target className="h-4 w-4" aria-hidden="true" />
            Practice table
          </div>
          <p className="mt-2 hidden max-w-2xl text-sm leading-6 text-muted sm:block">
            Play exact-card heads-up spots against a pinned baseline. Accepted strategy is labeled Approximate GTO; adaptive opponent responses are reported separately and never change your grading target.
          </p>
        </div>
        <Link
          href="/stats"
          className="inline-flex min-h-11 items-center gap-2 rounded-md border border-border bg-surface px-4 text-sm font-semibold hover:border-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          <BarChart3 className="h-4 w-4" aria-hidden="true" />
          Stats
        </Link>
      </header>

      {goalSummary && (
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-md border border-accent/35 bg-accent/10 px-4 py-3" role="status">
          <div className="flex items-start gap-2">
            <Info className="mt-0.5 h-4 w-4 text-accent" aria-hidden="true" />
            <div>
              <p className="text-sm font-semibold">Decision goal reached</p>
              <p className="mt-0.5 text-xs text-muted">The hand is complete. Continuing resumes on this same table.</p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setRailTab('stats')}
            className="min-h-11 rounded-md border border-accent px-3 text-sm font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            View run summary
          </button>
        </div>
      )}

      {pendingSettings && (
        <button
          type="button"
          onClick={() => setRailTab('settings')}
          className="mb-4 flex min-h-11 w-full items-center justify-between gap-3 rounded-md border border-accent/35 bg-accent/10 px-4 text-left text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          <span><strong>Pending settings:</strong> changes apply after this hand.</span>
          <Settings2 className="h-4 w-4 shrink-0" aria-hidden="true" />
        </button>
      )}

      <div className="grid min-w-0 gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">
        <PracticeTable
          state={state}
          node={spot?.node ?? activeNode}
          status={status}
          mode={settings.mode}
          modelLabel={modelLabel}
          unavailableMessage={unavailableCopy(settings)}
          errorMessage={errorMessage}
          revealOpponent={status === 'review'}
          selectedActionId={selectedActionId}
          onAction={(action) => void chooseAction(action)}
          onContinue={continueHand}
          onRetry={retry}
        />
        <div className="hidden xl:block">
          <AnalystRail
            idPrefix="desktop-rail"
            tab={railTab}
            onTabChange={setRailTab}
            feedback={feedback}
            recentHands={recentHands}
            settings={settings}
            pendingSettings={pendingSettings}
            onSettingsChange={updateSettings}
            fullDepths={depths}
            manifest={manifest}
            sessionDecisions={sessionDecisions}
            historyWarning={historyWarning}
            opponentModel={opponentModel}
          />
        </div>
      </div>

      <button
        type="button"
        aria-haspopup="dialog"
        aria-expanded={mobileRailOpen}
        onClick={() => setMobileRailOpen(true)}
        className="fixed bottom-[calc(8.8rem+env(safe-area-inset-bottom))] right-4 z-30 inline-flex min-h-11 items-center gap-2 rounded-full border border-border bg-surface px-4 text-sm font-semibold shadow-xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent xl:hidden"
      >
        <BarChart3 className="h-4 w-4 text-accent" aria-hidden="true" />
        Analyst
      </button>

      {mobileRailOpen && (
        <div className="fixed inset-0 z-50 bg-black/45 xl:hidden" role="presentation" onMouseDown={(event) => {
          if (event.target === event.currentTarget) setMobileRailOpen(false);
        }}>
          <section
            ref={mobileSheetRef}
            role="dialog"
            aria-modal="true"
            aria-label="Table analyst"
            className="absolute inset-x-0 bottom-0 max-h-[calc(100dvh-4rem)] overflow-hidden rounded-t-xl bg-bg pb-[env(safe-area-inset-bottom)] shadow-2xl"
          >
            <div className="flex min-h-12 items-center justify-between border-b border-border px-4">
              <p className="text-sm font-semibold">Table analyst</p>
              <button
                ref={mobileCloseRef}
                type="button"
                aria-label="Close analyst"
                onClick={() => setMobileRailOpen(false)}
                className="grid h-11 w-11 place-items-center rounded-md text-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
              >
                <X className="h-5 w-5" aria-hidden="true" />
              </button>
            </div>
            <AnalystRail
              idPrefix="mobile-rail"
              tab={railTab}
              onTabChange={setRailTab}
              feedback={feedback}
              recentHands={recentHands}
              settings={settings}
              pendingSettings={pendingSettings}
              onSettingsChange={updateSettings}
              fullDepths={depths}
              manifest={manifest}
              sessionDecisions={sessionDecisions}
              historyWarning={historyWarning}
              opponentModel={opponentModel}
            />
          </section>
        </div>
      )}
    </div>
  );
}
