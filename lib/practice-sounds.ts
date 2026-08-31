import type { ActionKind, HandState } from '@/lib/practice-types';

export interface PracticeSoundSnapshot {
  handId: string;
  boardCount: number;
  actionKinds: ActionKind[];
}

export type PracticeSoundCue =
  | { kind: 'cards'; count: number }
  | { kind: 'chips' };

interface PracticeSoundBuffers {
  cardSlide: AudioBuffer;
  chipLay: AudioBuffer;
  chipsCollide: AudioBuffer;
}

interface QueuedPracticeSound {
  cue: PracticeSoundCue;
  delaySeconds: number;
}

const WAGER_ACTIONS = new Set<ActionKind>([
  'call',
  'bet',
  'raise',
  'all-in',
]);

const SOUND_URLS = {
  cardSlide: '/sounds/card-slide-1.ogg',
  chipLay: '/sounds/chip-lay-1.ogg',
  chipsCollide: '/sounds/chips-collide-1.ogg',
} as const;

let audioContext: AudioContext | null = null;
let soundBuffers: PracticeSoundBuffers | null = null;
let soundBuffersPromise: Promise<PracticeSoundBuffers> | null = null;
let pendingSounds: QueuedPracticeSound[] = [];

export function practiceSoundSnapshot(
  state: HandState | null
): PracticeSoundSnapshot | null {
  if (!state) return null;
  return {
    handId: state.id,
    boardCount: state.board.length,
    actionKinds: state.actionHistory.map((action) => action.kind),
  };
}

export function practiceSoundCues(
  previous: PracticeSoundSnapshot | null,
  current: PracticeSoundSnapshot | null
): PracticeSoundCue[] {
  if (!current) return [];
  if (!previous || previous.handId !== current.handId) {
    const cues: PracticeSoundCue[] = [
      { kind: 'chips' },
      { kind: 'cards', count: 2 },
    ];
    if (current.boardCount > 0) {
      cues.push({ kind: 'cards', count: Math.min(current.boardCount, 3) });
    }
    return cues;
  }

  const cues: PracticeSoundCue[] = [];
  const newActions = current.actionKinds.slice(previous.actionKinds.length);
  if (newActions.some((kind) => WAGER_ACTIONS.has(kind))) {
    cues.push({ kind: 'chips' });
  }

  const dealtCards = current.boardCount - previous.boardCount;
  if (dealtCards > 0) {
    cues.push({ kind: 'cards', count: Math.min(dealtCards, 3) });
  }
  return cues;
}

function getAudioContext(): AudioContext | null {
  if (typeof window === 'undefined') return null;
  if (!audioContext || audioContext.state === 'closed') {
    try {
      audioContext = new window.AudioContext({ latencyHint: 'interactive' });
      soundBuffers = null;
      soundBuffersPromise = null;
    } catch {
      return null;
    }
  }
  return audioContext;
}

async function decodeSound(
  context: AudioContext,
  url: string
): Promise<AudioBuffer> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Unable to load practice sound: ${url}`);
  }
  return context.decodeAudioData(await response.arrayBuffer());
}

function loadSoundBuffers(
  context: AudioContext
): Promise<PracticeSoundBuffers> {
  if (soundBuffers) return Promise.resolve(soundBuffers);
  if (!soundBuffersPromise) {
    soundBuffersPromise = Promise.all([
      decodeSound(context, SOUND_URLS.cardSlide),
      decodeSound(context, SOUND_URLS.chipLay),
      decodeSound(context, SOUND_URLS.chipsCollide),
    ])
      .then(([cardSlide, chipLay, chipsCollide]) => {
        soundBuffers = { cardSlide, chipLay, chipsCollide };
        return soundBuffers;
      })
      .catch((error: unknown) => {
        soundBuffersPromise = null;
        throw error;
      });
  }
  return soundBuffersPromise;
}

export function preloadPracticeSounds(): void {
  const context = getAudioContext();
  if (!context) return;
  void loadSoundBuffers(context).catch(() => undefined);
}

function scheduleSample(
  context: AudioContext,
  buffer: AudioBuffer,
  start: number,
  gainValue: number,
  playbackRate: number,
  metallic = false
): void {
  const source = context.createBufferSource();
  const gain = context.createGain();
  source.buffer = buffer;
  source.playbackRate.setValueAtTime(playbackRate, start);
  gain.gain.setValueAtTime(gainValue, start);

  if (metallic) {
    const sheen = context.createBiquadFilter();
    sheen.type = 'highshelf';
    sheen.frequency.setValueAtTime(2400, start);
    sheen.gain.setValueAtTime(4, start);
    source.connect(sheen);
    sheen.connect(gain);
  } else {
    source.connect(gain);
  }

  gain.connect(context.destination);
  source.start(start);
}

function playCardDeal(
  context: AudioContext,
  buffers: PracticeSoundBuffers,
  count: number,
  delaySeconds: number
): void {
  const cardCount = Math.max(1, Math.min(count, 3));
  const rates = [0.94, 1.04, 0.99];
  for (let index = 0; index < cardCount; index++) {
    scheduleSample(
      context,
      buffers.cardSlide,
      context.currentTime + delaySeconds + index * 0.09,
      0.42,
      rates[index]
    );
  }
}

function playChipStack(
  context: AudioContext,
  buffers: PracticeSoundBuffers,
  delaySeconds: number
): void {
  const start = context.currentTime + delaySeconds;
  scheduleSample(context, buffers.chipLay, start, 0.48, 0.97);
  scheduleSample(context, buffers.chipsCollide, start + 0.025, 0.38, 0.94, true);
  scheduleSample(context, buffers.chipsCollide, start + 0.075, 0.23, 1.08, true);
}

function playLoadedPracticeSound(
  context: AudioContext,
  cue: PracticeSoundCue,
  delaySeconds: number
): boolean {
  if (context.state !== 'running' || !soundBuffers) return false;
  try {
    if (cue.kind === 'cards') {
      playCardDeal(context, soundBuffers, cue.count, delaySeconds);
    } else {
      playChipStack(context, soundBuffers, delaySeconds);
    }
  } catch {
    return false;
  }
  return true;
}

function flushPendingSounds(context: AudioContext): void {
  if (context.state !== 'running' || !soundBuffers || pendingSounds.length === 0) {
    return;
  }

  const queued = pendingSounds;
  pendingSounds = [];
  let cursor = 0;
  queued.forEach(({ cue, delaySeconds }) => {
    const scheduledDelay = Math.max(delaySeconds, cursor);
    playLoadedPracticeSound(context, cue, scheduledDelay);
    cursor = scheduledDelay + (cue.kind === 'cards' ? cue.count * 0.09 : 0.13);
  });
}

export function clearPendingPracticeSounds(): void {
  pendingSounds = [];
}

export async function unlockPracticeAudio(): Promise<boolean> {
  const context = getAudioContext();
  if (!context) return false;
  const loading = loadSoundBuffers(context);
  if (context.state !== 'running') {
    try {
      await context.resume();
    } catch {
      return false;
    }
  }
  try {
    await loading;
  } catch {
    return false;
  }
  flushPendingSounds(context);
  return context.state === 'running';
}

export function playPracticeSound(
  cue: PracticeSoundCue,
  delaySeconds = 0
): boolean {
  const context = getAudioContext();
  if (!context || context.state !== 'running' || !soundBuffers) {
    pendingSounds = [...pendingSounds.slice(-7), { cue, delaySeconds }];
    if (context) {
      void loadSoundBuffers(context)
        .then(() => flushPendingSounds(context))
        .catch(() => undefined);
    }
    return false;
  }
  return playLoadedPracticeSound(context, cue, delaySeconds);
}
