'use client';

import Link from 'next/link';
import { useMemo, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  BarChart3,
  Check,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Play,
  RotateCcw,
  Settings2,
  SlidersHorizontal,
  Target,
  X,
  XCircle,
} from 'lucide-react';
import {
  chartsForPractice,
  DEFAULT_PRACTICE_RULES,
  generatePracticeQuestions,
  recordPracticeAnswer,
  type PracticeAction,
  type PracticeCategory,
  type PracticeQuestion,
  type PracticeRecord,
  type PracticeRules,
} from '@/lib/practice';
import { appendPracticeRecords } from '@/lib/practice-history';
import {
  defaultScenarioForSeats,
  openingSizeLabel,
  scenariosForSeats,
} from '@/data/preflop/catalog';
import {
  formatForSeats,
  positionFullForSeats,
  positionLabelForSeats,
  TABLE_FORMATS,
  type Position,
  type TableSeats,
} from '@/lib/positions';

type View = 'setup' | 'session' | 'complete';

const QUESTION_COUNTS = [10, 20, 30] as const;

const CATEGORY_OPTIONS: Array<{
  value: PracticeCategory;
  label: string;
  description: string;
}> = [
  {
    value: 'RFI',
    label: 'Raise first in',
    description: 'The action folds to you.',
  },
  {
    value: 'vs-RFI',
    label: 'Facing a raise',
    description: 'Respond to a single open.',
  },
];

const ACTION_STYLES: Record<PracticeAction, string> = {
  Fold: 'border-fold/50 hover:border-fold hover:bg-fold/10',
  Call: 'border-call/50 hover:border-call hover:bg-call/10',
  Raise: 'border-raise/50 hover:border-raise hover:bg-raise/10',
  '3-bet': 'border-allin/50 hover:border-allin hover:bg-allin/10',
  'All-in': 'border-allin/50 hover:border-allin hover:bg-allin/10',
};

const ACTION_FILLED_STYLES: Record<PracticeAction, string> = {
  Fold: 'border-fold bg-fold text-fold-fg',
  Call: 'border-call bg-call text-call-fg',
  Raise: 'border-raise bg-raise text-raise-fg',
  '3-bet': 'border-allin bg-allin text-allin-fg',
  'All-in': 'border-allin bg-allin text-allin-fg',
};

function toggleValue<T>(values: T[], value: T): T[] {
  return values.includes(value)
    ? values.filter((candidate) => candidate !== value)
    : [...values, value];
}

function handType(handClass: string): string {
  if (handClass.length === 2) return 'Pocket pair';
  return handClass.endsWith('s') ? 'Suited' : 'Offsuit';
}

function percentage(value: number): string {
  const percent = value * 100;
  return Number.isInteger(percent)
    ? `${percent.toFixed(0)}%`
    : `${percent.toFixed(1)}%`;
}

function categoryCopy(category: PracticeCategory, allIn: boolean) {
  if (category === 'RFI') {
    return allIn
      ? {
          label: 'Push or fold',
          description: 'Choose whether to move all-in when first to act.',
        }
      : CATEGORY_OPTIONS[0];
  }
  return allIn
    ? {
        label: 'Facing a shove',
        description: 'Choose whether to call an all-in.',
      }
    : CATEGORY_OPTIONS[1];
}

function actionColor(action: PracticeAction): string {
  switch (action) {
    case 'Fold':
      return 'bg-fold';
    case 'Call':
      return 'bg-call';
    case 'Raise':
      return 'bg-raise';
    case '3-bet':
    case 'All-in':
      return 'bg-allin';
  }
}

export default function PracticePage() {
  const [view, setView] = useState<View>('setup');
  const [rules, setRules] = useState<PracticeRules>(DEFAULT_PRACTICE_RULES);
  const [questions, setQuestions] = useState<PracticeQuestion[]>([]);
  const [questionIndex, setQuestionIndex] = useState(0);
  const [selectedAction, setSelectedAction] =
    useState<PracticeAction | null>(null);
  const [records, setRecords] = useState<PracticeRecord[]>([]);
  const [setupError, setSetupError] = useState('');
  const [historyWarning, setHistoryWarning] = useState('');
  const questionStartedAt = useRef(0);

  const availableCharts = useMemo(() => chartsForPractice(rules), [rules]);
  const scenarios = useMemo(
    () => scenariosForSeats(rules.seats),
    [rules.seats]
  );
  const activeScenario =
    scenarios.find((scenario) => scenario.id === rules.scenarioId) ??
    scenarios[0];
  const currentQuestion = questions[questionIndex];
  const answeredCount = records.length;
  const correctCount = records.filter((record) => record.correct).length;
  const sessionAccuracy = answeredCount
    ? Math.round((correctCount / answeredCount) * 100)
    : 0;

  function setSeats(seats: TableSeats) {
    const scenario = defaultScenarioForSeats(seats);
    setRules((current) => ({
      ...current,
      seats,
      scenarioId: scenario.id,
      positions: [...formatForSeats(seats).positions],
    }));
    setSetupError('');
  }

  function setScenario(scenarioId: string) {
    setRules((current) => ({ ...current, scenarioId }));
    setSetupError('');
  }

  function toggleCategory(category: PracticeCategory) {
    setRules((current) => {
      if (
        current.categories.includes(category) &&
        current.categories.length === 1
      ) {
        return current;
      }
      return {
        ...current,
        categories: toggleValue(current.categories, category),
      };
    });
    setSetupError('');
  }

  function togglePosition(position: Position) {
    setRules((current) => {
      if (
        current.positions.includes(position) &&
        current.positions.length === 1
      ) {
        return current;
      }
      return {
        ...current,
        positions: toggleValue(current.positions, position),
      };
    });
    setSetupError('');
  }

  function beginSession(event?: FormEvent) {
    event?.preventDefault();
    const nextQuestions = generatePracticeQuestions(rules);
    if (nextQuestions.length === 0) {
      setSetupError(
        'No charts match these rules. Add a compatible spot type or position.'
      );
      return;
    }

    setQuestions(nextQuestions);
    setQuestionIndex(0);
    setSelectedAction(null);
    setRecords([]);
    setSetupError('');
    setHistoryWarning('');
    questionStartedAt.current = performance.now();
    window.scrollTo(0, 0);
    setView('session');
  }

  function answerQuestion(action: PracticeAction) {
    if (!currentQuestion || selectedAction) return;
    const record = recordPracticeAnswer(
      currentQuestion,
      action,
      performance.now() - questionStartedAt.current
    );
    const saved = appendPracticeRecords([record]);
    if (!saved) {
      setHistoryWarning(
        'This answer is counted in the session but could not be saved to this device.'
      );
    }
    setRecords((current) => [...current, record]);
    setSelectedAction(action);
  }

  function advanceQuestion() {
    if (!selectedAction) return;
    if (questionIndex >= questions.length - 1) {
      window.scrollTo(0, 0);
      setView('complete');
      return;
    }

    setQuestionIndex((current) => current + 1);
    setSelectedAction(null);
    questionStartedAt.current = performance.now();
  }

  function returnToSetup() {
    window.scrollTo(0, 0);
    setView('setup');
    setQuestions([]);
    setRecords([]);
    setQuestionIndex(0);
    setSelectedAction(null);
    setSetupError('');
    setHistoryWarning('');
  }

  if (view === 'setup') {
    return (
      <div className="mx-auto w-full max-w-5xl pb-10">
        <header className="border-b border-border pb-6">
          <div className="flex items-center gap-2 font-mono text-xs font-semibold uppercase text-accent">
            <Target aria-hidden="true" className="h-4 w-4" />
            Practice
          </div>
          <h1 className="mt-3 text-3xl font-semibold leading-tight sm:text-4xl">
            Build your session
          </h1>
          <p className="mt-3 max-w-2xl text-base leading-7 text-muted">
            Choose the table, spots, and positions you want to train. Each
            session samples real hand classes from the preflop library.
          </p>
        </header>

        <div className="mt-7 grid gap-8 lg:grid-cols-[minmax(0,1fr)_300px]">
          <form
            onSubmit={beginSession}
            className="rounded-lg border border-border bg-surface shadow-sm"
          >
            <fieldset className="border-b border-border p-5 sm:p-6">
              <legend className="flex items-center gap-2 text-sm font-semibold">
                <span className="grid h-7 w-7 place-items-center rounded-md bg-surface-2 font-mono text-xs text-muted">
                  1
                </span>
                Table format
              </legend>
              <div
                className="mt-4 grid grid-cols-3 gap-1 rounded-lg border border-border bg-surface-2 p-1"
                aria-label="Table format"
              >
                {TABLE_FORMATS.map((format) => (
                  <button
                    key={format.seats}
                    type="button"
                    aria-pressed={rules.seats === format.seats}
                    onClick={() => setSeats(format.seats)}
                    className={
                      'min-h-11 cursor-pointer rounded-md px-3 text-sm font-semibold transition-[background-color,color,box-shadow] duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ' +
                      (rules.seats === format.seats
                        ? 'bg-fg text-bg shadow-sm'
                        : 'text-muted hover:bg-surface hover:text-fg')
                    }
                  >
                    {format.label}
                  </button>
                ))}
              </div>
            </fieldset>

            <fieldset className="border-b border-border p-5 sm:p-6">
              <legend className="flex items-center gap-2 text-sm font-semibold">
                <span className="grid h-7 w-7 place-items-center rounded-md bg-surface-2 font-mono text-xs text-muted">
                  2
                </span>
                Range source and stack
              </legend>
              <div className="mt-4 grid gap-2 sm:grid-cols-2">
                {scenarios.map((scenario) => {
                  const selected = scenario.id === activeScenario?.id;
                  return (
                    <button
                      key={scenario.id}
                      type="button"
                      aria-pressed={selected}
                      onClick={() => setScenario(scenario.id)}
                      className={
                        'min-h-16 rounded-md border px-4 py-3 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ' +
                        (selected
                          ? 'border-accent bg-accent/10'
                          : 'border-border hover:border-accent/60 hover:bg-surface-2')
                      }
                    >
                      <span className="block text-sm font-semibold text-fg">
                        {scenario.effectiveStackBb}bb · {scenario.label}
                      </span>
                      <span className="mt-1 block text-xs text-muted">
                        {openingSizeLabel(scenario.openingSize)}
                      </span>
                    </button>
                  );
                })}
              </div>
            </fieldset>

            <fieldset className="border-b border-border p-5 sm:p-6">
              <legend className="flex items-center gap-2 text-sm font-semibold">
                <span className="grid h-7 w-7 place-items-center rounded-md bg-surface-2 font-mono text-xs text-muted">
                  3
                </span>
                Spot types
              </legend>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                {CATEGORY_OPTIONS.map((category) => {
                  const selected = rules.categories.includes(category.value);
                  const copy = categoryCopy(
                    category.value,
                    activeScenario?.openingSize.kind === 'all-in'
                  );
                  return (
                    <button
                      key={category.value}
                      type="button"
                      aria-pressed={selected}
                      onClick={() => toggleCategory(category.value)}
                      className={
                        'flex min-h-20 cursor-pointer items-start gap-3 rounded-lg border p-4 text-left transition-[background-color,border-color] duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ' +
                        (selected
                          ? 'border-accent bg-accent/10'
                          : 'border-border hover:border-accent/60 hover:bg-surface-2')
                      }
                    >
                      <span
                        className={
                          'mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded border ' +
                          (selected
                            ? 'border-accent bg-accent text-accent-fg'
                            : 'border-border bg-surface')
                        }
                        aria-hidden="true"
                      >
                        {selected && <Check className="h-3.5 w-3.5" />}
                      </span>
                      <span>
                        <span className="block text-sm font-semibold">
                          {copy.label}
                        </span>
                        <span className="mt-1 block text-xs leading-5 text-muted">
                          {copy.description}
                        </span>
                      </span>
                    </button>
                  );
                })}
              </div>
            </fieldset>

            <fieldset className="border-b border-border p-5 sm:p-6">
              <legend className="flex items-center gap-2 text-sm font-semibold">
                <span className="grid h-7 w-7 place-items-center rounded-md bg-surface-2 font-mono text-xs text-muted">
                  4
                </span>
                Eligible positions
              </legend>
              <div className="mt-4 flex flex-wrap gap-2">
                {formatForSeats(rules.seats).positions.map((position) => {
                  const selected = rules.positions.includes(position);
                  return (
                    <button
                      key={position}
                      type="button"
                      aria-pressed={selected}
                      onClick={() => togglePosition(position)}
                      title={positionFullForSeats(position, rules.seats)}
                      className={
                        'min-h-11 min-w-14 cursor-pointer rounded-md border px-3 text-sm font-mono font-semibold transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ' +
                        (selected
                          ? 'border-accent bg-accent text-accent-fg'
                          : 'border-border bg-surface text-muted hover:border-accent hover:text-fg')
                      }
                    >
                      {positionLabelForSeats(position, rules.seats)}
                    </button>
                  );
                })}
              </div>
            </fieldset>

            <fieldset className="p-5 sm:p-6">
              <legend className="flex items-center gap-2 text-sm font-semibold">
                <span className="grid h-7 w-7 place-items-center rounded-md bg-surface-2 font-mono text-xs text-muted">
                  5
                </span>
                Session length
              </legend>
              <div className="mt-4 grid grid-cols-3 gap-2">
                {QUESTION_COUNTS.map((count) => (
                  <button
                    key={count}
                    type="button"
                    aria-pressed={rules.questionCount === count}
                    onClick={() =>
                      setRules((current) => ({
                        ...current,
                        questionCount: count,
                      }))
                    }
                    className={
                      'min-h-12 cursor-pointer rounded-md border px-3 text-sm font-semibold transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ' +
                      (rules.questionCount === count
                        ? 'border-fg bg-fg text-bg'
                        : 'border-border text-muted hover:border-fg hover:text-fg')
                    }
                  >
                    <span className="font-mono text-base">{count}</span>{' '}
                    questions
                  </button>
                ))}
              </div>
            </fieldset>

            <div className="border-t border-border bg-surface-2 p-5 sm:flex sm:items-center sm:justify-between sm:gap-5 sm:p-6">
              <div>
                <p className="text-sm font-medium">
                  {availableCharts.length}{' '}
                  {availableCharts.length === 1 ? 'chart' : 'charts'} in this
                  session pool
                </p>
                {setupError && (
                  <p className="mt-1 text-sm text-raise" role="alert">
                    {setupError}
                  </p>
                )}
              </div>
              <button
                type="submit"
                disabled={availableCharts.length === 0}
                className="mt-4 inline-flex min-h-12 w-full cursor-pointer items-center justify-center gap-2 rounded-lg bg-accent px-6 py-3 text-sm font-semibold text-accent-fg shadow-sm transition-[opacity,box-shadow] duration-200 hover:opacity-90 hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface-2 disabled:cursor-not-allowed disabled:opacity-40 sm:mt-0 sm:w-auto"
              >
                <Play aria-hidden="true" className="h-4 w-4" />
                Start session
              </button>
            </div>
          </form>

          <aside className="lg:pt-1" aria-labelledby="session-summary-title">
            <div className="flex items-center gap-2">
              <SlidersHorizontal
                aria-hidden="true"
                className="h-4 w-4 text-accent"
              />
              <h2 id="session-summary-title" className="text-sm font-semibold">
                Session rules
              </h2>
            </div>
            <dl className="mt-4 divide-y divide-border border-y border-border">
              <div className="flex items-center justify-between gap-4 py-4">
                <dt className="text-sm text-muted">Table</dt>
                <dd className="text-sm font-semibold">
                  {formatForSeats(rules.seats).label}
                </dd>
              </div>
              <div className="flex items-center justify-between gap-4 py-4">
                <dt className="text-sm text-muted">Ranges</dt>
                <dd className="text-right text-sm font-semibold">
                  {activeScenario?.effectiveStackBb}bb · {activeScenario?.label}
                </dd>
              </div>
              <div className="flex items-center justify-between gap-4 py-4">
                <dt className="text-sm text-muted">Spot types</dt>
                <dd className="text-right text-sm font-semibold">
                  {rules.categories.length === 2
                    ? 'Both'
                    : categoryCopy(
                        rules.categories[0],
                        activeScenario?.openingSize.kind === 'all-in'
                      ).label}
                </dd>
              </div>
              <div className="flex items-center justify-between gap-4 py-4">
                <dt className="text-sm text-muted">Positions</dt>
                <dd className="font-mono text-sm font-semibold">
                  {rules.positions.length}/
                  {formatForSeats(rules.seats).positions.length}
                </dd>
              </div>
              <div className="flex items-center justify-between gap-4 py-4">
                <dt className="text-sm text-muted">Questions</dt>
                <dd className="font-mono text-sm font-semibold">
                  {rules.questionCount}
                </dd>
              </div>
            </dl>
            <div className="mt-6 flex gap-3 text-sm leading-6 text-muted">
              <Settings2
                aria-hidden="true"
                className="mt-0.5 h-4 w-4 shrink-0 text-accent"
              />
              <p>
                Your choices stay local. Completed decisions are saved
                automatically for your stats.
              </p>
            </div>
          </aside>
        </div>
      </div>
    );
  }

  if (view === 'complete') {
    const averageResponseMs = records.length
      ? Math.round(
          records.reduce((sum, record) => sum + record.responseMs, 0) /
            records.length
        )
      : 0;

    return (
      <div className="mx-auto flex min-h-[calc(100svh-8rem)] w-full max-w-3xl items-center py-8">
        <section
          aria-labelledby="session-complete-title"
          className="w-full rounded-lg border border-border bg-surface p-6 shadow-sm sm:p-10"
        >
          <span className="grid h-12 w-12 place-items-center rounded-lg bg-call/15 text-call">
            <CheckCircle2 aria-hidden="true" className="h-6 w-6" />
          </span>
          <p className="mt-6 font-mono text-xs font-semibold uppercase text-accent">
            Session complete
          </p>
          <h1
            id="session-complete-title"
            className="mt-2 text-3xl font-semibold leading-tight sm:text-4xl"
          >
            {sessionAccuracy}% decision accuracy
          </h1>
          <p className="mt-3 max-w-xl text-base leading-7 text-muted">
            You made {correctCount} correct decisions across {records.length}{' '}
            spots.{' '}
            {historyWarning
              ? 'This session includes answers that could not be saved.'
              : 'Every answer is now included in your practice history.'}
          </p>

          <dl className="mt-8 grid gap-px overflow-hidden rounded-lg border border-border bg-border sm:grid-cols-3">
            <div className="bg-surface-2 p-4">
              <dt className="text-xs font-medium text-muted">Correct</dt>
              <dd className="mt-1 font-mono text-2xl font-semibold">
                {correctCount}/{records.length}
              </dd>
            </div>
            <div className="bg-surface-2 p-4">
              <dt className="text-xs font-medium text-muted">Average time</dt>
              <dd className="mt-1 font-mono text-2xl font-semibold">
                {(averageResponseMs / 1000).toFixed(1)}s
              </dd>
            </div>
            <div className="bg-surface-2 p-4">
              <dt className="text-xs font-medium text-muted">Format</dt>
              <dd className="mt-1 text-lg font-semibold">
                {formatForSeats(rules.seats).label}
              </dd>
            </div>
          </dl>

          <div className="mt-8 flex flex-col gap-3 sm:flex-row">
            <button
              type="button"
              onClick={() => beginSession()}
              className="inline-flex min-h-12 cursor-pointer items-center justify-center gap-2 rounded-lg bg-accent px-6 py-3 text-sm font-semibold text-accent-fg transition-opacity duration-200 hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface"
            >
              <RotateCcw aria-hidden="true" className="h-4 w-4" />
              Practice again
            </button>
            <Link
              href="/stats"
              className="inline-flex min-h-12 cursor-pointer items-center justify-center gap-2 rounded-lg border border-border px-6 py-3 text-sm font-semibold transition-[border-color,color] duration-200 hover:border-accent hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface"
            >
              <BarChart3 aria-hidden="true" className="h-4 w-4" />
              View strength report
            </Link>
          </div>
          <button
            type="button"
            onClick={returnToSetup}
            className="mt-5 inline-flex min-h-11 cursor-pointer items-center gap-2 text-sm font-medium text-muted transition-colors duration-200 hover:text-fg focus-visible:rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            <ArrowLeft aria-hidden="true" className="h-4 w-4" />
            Change session rules
          </button>
        </section>
      </div>
    );
  }

  if (!currentQuestion) return null;

  const selectedRecord = records[questionIndex];
  const responseSeconds = selectedRecord
    ? (selectedRecord.responseMs / 1000).toFixed(1)
    : null;
  const progress = ((questionIndex + (selectedAction ? 1 : 0)) / questions.length) *
    100;
  const openingSize = currentQuestion.scenario.openingSize;
  const prompt =
    currentQuestion.category === 'RFI'
      ? openingSize.kind === 'all-in'
        ? 'The action is on you. Choose between folding and moving all-in.'
        : currentQuestion.seats === 2
          ? 'You are first to act before the flop.'
          : 'The action folds to you.'
      : openingSize.kind === 'all-in'
        ? `${currentQuestion.villain ?? 'Villain'} moves all-in for ${currentQuestion.scenario.effectiveStackBb}bb.`
        : openingSize.kind === 'raise-to'
          ? `${currentQuestion.villain ?? 'Villain'} opens to ${openingSize.bb}bb.`
          : `${currentQuestion.villain ?? 'Villain'} makes the reference opening action.`;
  const heroLabel =
    currentQuestion.seats === 2 && currentQuestion.hero === 'BTN'
      ? 'Button / small blind'
      : positionFullForSeats(
          currentQuestion.hero,
          currentQuestion.seats
        );

  return (
    <div className="mx-auto w-full max-w-5xl pb-10">
      <header className="border-b border-border pb-5">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="font-mono text-xs font-semibold uppercase text-accent">
              Practice session
            </p>
            <h1 className="mt-2 text-2xl font-semibold">
              Question {questionIndex + 1} of {questions.length}
            </h1>
          </div>
          <button
            type="button"
            onClick={returnToSetup}
            className="inline-flex min-h-11 cursor-pointer items-center gap-2 rounded-md px-3 text-sm font-medium text-muted transition-colors duration-200 hover:bg-surface-2 hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            <Settings2 aria-hidden="true" className="h-4 w-4" />
            Session rules
          </button>
        </div>
        <div
          className="mt-5 h-1.5 overflow-hidden rounded-sm bg-surface-2"
          role="progressbar"
          aria-label="Session progress"
          aria-valuemin={0}
          aria-valuemax={questions.length}
          aria-valuenow={questionIndex + (selectedAction ? 1 : 0)}
        >
          <div
            className="h-full bg-accent transition-[width] duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      </header>

      <div className="mt-6 grid gap-6 lg:grid-cols-[minmax(0,1fr)_260px]">
        <section className="overflow-hidden rounded-lg border border-border bg-surface shadow-sm">
          <div className="border-b border-border bg-surface-2 px-5 py-4 sm:px-7">
            <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
              <span className="font-semibold">
                {formatForSeats(currentQuestion.seats).label}
              </span>
              <span className="text-muted">
                {currentQuestion.scenario.effectiveStackBb}bb effective
              </span>
              <span className="text-muted">
                {currentQuestion.category === 'RFI'
                  ? openingSize.kind === 'all-in'
                    ? 'Push or fold'
                    : 'Raise first in'
                  : openingSize.kind === 'all-in'
                    ? 'Facing a shove'
                    : 'Facing an open'}
              </span>
            </div>
          </div>

          <div className="px-5 py-7 sm:px-7 sm:py-8">
            <div className="flex flex-col items-center text-center">
              <p className="text-sm text-muted">{prompt}</p>
              <h2 className="mt-2 text-xl font-semibold sm:text-2xl">
                You are {heroLabel}. What is your action?
              </h2>

              <div className="mt-7 flex items-end justify-center gap-3">
                {currentQuestion.handClass.slice(0, 2).split('').map((rank, index) => (
                  <div
                    key={`${rank}-${index}`}
                    className={
                      'grid h-28 w-20 place-items-center rounded-lg border bg-bg shadow-sm sm:h-32 sm:w-24 ' +
                      (index === 0
                        ? 'border-fg/30 text-fg'
                        : 'border-raise/40 text-raise')
                    }
                    aria-hidden="true"
                  >
                    <span className="font-mono text-4xl font-semibold">
                      {rank}
                    </span>
                  </div>
                ))}
              </div>
              <p className="mt-3 font-mono text-sm font-semibold">
                {currentQuestion.handClass}{' '}
                <span className="font-sans font-normal text-muted">
                  · {handType(currentQuestion.handClass)}
                </span>
              </p>
            </div>

            <div className="mt-8 border-t border-border pt-6">
              <p className="mb-3 text-sm font-semibold">Choose an action</p>
              <div
                className={
                  'grid gap-3 ' +
                  (currentQuestion.options.length === 2
                    ? 'grid-cols-2'
                    : 'grid-cols-1 sm:grid-cols-3')
                }
              >
                {currentQuestion.options.map((action) => {
                  const chosen = selectedAction === action;
                  const recommended =
                    selectedAction &&
                    currentQuestion.correctActions.includes(action);
                  return (
                    <button
                      key={action}
                      type="button"
                      disabled={selectedAction !== null}
                      onClick={() => answerQuestion(action)}
                      aria-label={
                        selectedAction
                          ? `${action}${chosen ? ', your answer' : ''}${recommended ? ', valid chart action' : ''}`
                          : action
                      }
                      className={
                        'relative min-h-14 cursor-pointer rounded-lg border px-4 py-3 text-sm font-semibold transition-[background-color,border-color,color] duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface disabled:cursor-default disabled:opacity-100 ' +
                        (chosen
                          ? ACTION_FILLED_STYLES[action]
                          : recommended
                            ? 'border-call bg-call/10 text-fg'
                            : ACTION_STYLES[action])
                      }
                    >
                      {action}
                      {recommended && !chosen && (
                        <Check
                          aria-hidden="true"
                          className="absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-call"
                        />
                      )}
                    </button>
                  );
                })}
              </div>
            </div>

            <div
              className="mt-6 min-h-[210px] border-t border-border pt-6"
              aria-live="polite"
            >
              {selectedAction && selectedRecord ? (
                <>
                  <div className="flex items-start gap-3">
                    {selectedRecord.correct ? (
                      <CheckCircle2
                        aria-hidden="true"
                        className="mt-0.5 h-5 w-5 shrink-0 text-call"
                      />
                    ) : (
                      <XCircle
                        aria-hidden="true"
                        className="mt-0.5 h-5 w-5 shrink-0 text-raise"
                      />
                    )}
                    <div>
                      <h3 className="font-semibold">
                        {selectedRecord.correct
                          ? 'Correct decision'
                          : `Best default: ${currentQuestion.recommendedAction}`}
                      </h3>
                      <p className="mt-1 text-sm leading-6 text-muted">
                        Full strategy for {currentQuestion.handClass} in this
                        spot:
                      </p>
                    </div>
                  </div>

                  <div className="mt-5 grid gap-3 sm:grid-cols-3">
                    {currentQuestion.strategy.map((item) => (
                      <div key={item.action}>
                        <div className="mb-2 flex items-center justify-between gap-3 text-sm">
                          <span className="font-medium">{item.action}</span>
                          <span className="font-mono text-muted">
                            {percentage(item.frequency)}
                          </span>
                        </div>
                        <div className="h-2 overflow-hidden rounded-sm bg-surface-2">
                          <div
                            className={`h-full ${actionColor(item.action)}`}
                            style={{ width: percentage(item.frequency) }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="mt-6 flex flex-col-reverse gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <span className="flex items-center gap-2 text-xs text-muted">
                      <Clock3 aria-hidden="true" className="h-4 w-4" />
                      Answered in {responseSeconds}s
                    </span>
                    <button
                      type="button"
                      onClick={advanceQuestion}
                      className="inline-flex min-h-12 cursor-pointer items-center justify-center gap-2 rounded-lg bg-accent px-6 py-3 text-sm font-semibold text-accent-fg transition-opacity duration-200 hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface"
                    >
                      {questionIndex === questions.length - 1
                        ? 'Finish session'
                        : 'Next question'}
                      <ArrowRight aria-hidden="true" className="h-4 w-4" />
                    </button>
                  </div>
                </>
              ) : (
                <div className="grid min-h-[170px] place-items-center text-center">
                  <div>
                    <Target
                      aria-hidden="true"
                      className="mx-auto h-5 w-5 text-muted"
                    />
                    <p className="mt-3 text-sm font-medium">
                      Commit to your decision.
                    </p>
                    <p className="mt-1 text-sm text-muted">
                      Strategy frequencies appear after you answer.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
          {historyWarning && (
            <p
              className="border-t border-raise/30 bg-raise/10 px-5 py-3 text-sm text-raise sm:px-7"
              role="status"
            >
              {historyWarning}
            </p>
          )}
        </section>

        <aside aria-labelledby="session-status-title">
          <div className="flex items-center gap-2">
            <BarChart3
              aria-hidden="true"
              className="h-4 w-4 text-accent"
            />
            <h2 id="session-status-title" className="text-sm font-semibold">
              Session status
            </h2>
          </div>
          <dl className="mt-4 divide-y divide-border border-y border-border">
            <div className="flex items-center justify-between py-4">
              <dt className="text-sm text-muted">Answered</dt>
              <dd className="font-mono text-sm font-semibold">
                {answeredCount}/{questions.length}
              </dd>
            </div>
            <div className="flex items-center justify-between py-4">
              <dt className="text-sm text-muted">Correct</dt>
              <dd className="font-mono text-sm font-semibold">
                {correctCount}
              </dd>
            </div>
            <div className="flex items-center justify-between py-4">
              <dt className="text-sm text-muted">Accuracy</dt>
              <dd className="font-mono text-sm font-semibold">
                {answeredCount ? `${sessionAccuracy}%` : '0%'}
              </dd>
            </div>
          </dl>

          <ol
            className="mt-5 grid grid-cols-10 gap-1.5 lg:grid-cols-5"
            aria-label="Question results"
          >
            {questions.map((question, index) => {
              const result = records[index];
              const isCurrent = index === questionIndex;
              return (
                <li
                  key={question.id}
                  aria-label={
                    result
                      ? `Question ${index + 1}, ${result.correct ? 'correct' : 'incorrect'}`
                      : `Question ${index + 1}${isCurrent ? ', current' : ''}`
                  }
                  className={
                    'grid aspect-square min-w-0 place-items-center rounded-md border font-mono text-xs ' +
                    (result
                      ? result.correct
                        ? 'border-call/50 bg-call/10 text-call'
                        : 'border-raise/50 bg-raise/10 text-raise'
                      : isCurrent
                        ? 'border-accent bg-accent text-accent-fg'
                        : 'border-border text-muted')
                  }
                >
                  {result ? (
                    result.correct ? (
                      <Check aria-hidden="true" className="h-3.5 w-3.5" />
                    ) : (
                      <X aria-hidden="true" className="h-3.5 w-3.5" />
                    )
                  ) : (
                    index + 1
                  )}
                </li>
              );
            })}
          </ol>

          <div className="mt-6 flex items-start gap-3 border-t border-border pt-5 text-sm leading-6 text-muted">
            <ChevronRight
              aria-hidden="true"
              className="mt-0.5 h-4 w-4 shrink-0 text-accent"
            />
            <p>
              {currentQuestion.category === 'RFI'
                ? 'Opening discipline'
                : `${currentQuestion.hero} versus ${currentQuestion.villain} defense`}
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}
