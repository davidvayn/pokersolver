import { NextResponse } from 'next/server';
import { modelForFullDepth } from '@/lib/practice-models';
import type { HandState } from '@/lib/practice-types';
import { resolverQueryPayload } from '@/lib/server/practice-resolver-request';
import {
  PRACTICE_RESOLVER_IDENTITY,
  practiceSolverProcess,
} from '@/lib/server/practice-solver-process';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function isCardPair(value: unknown): value is [number, number] {
  return (
    Array.isArray(value) &&
    value.length === 2 &&
    value.every((card) => Number.isInteger(card) && card >= 0 && card < 52) &&
    value[0] !== value[1]
  );
}

function isQueryState(value: unknown): value is HandState {
  if (!value || typeof value !== 'object') return false;
  const state = value as Partial<HandState>;
  return Boolean(
    typeof state.modelVersion === 'string' &&
      Number.isFinite(state.depthBb) &&
      ['preflop', 'flop', 'turn', 'river'].includes(state.street ?? '') &&
      state.toAct &&
      ['button-small-blind', 'big-blind'].includes(state.toAct) &&
      state.terminal === false &&
      state.holeCards &&
      isCardPair(state.holeCards[state.toAct]) &&
      Array.isArray(state.board) &&
      state.board.every(
        (card) => Number.isInteger(card) && card >= 0 && card < 52
      ) &&
      state.stacksBb &&
      state.streetBetsBb &&
      state.totalCommittedBb &&
      Number.isFinite(state.lastFullRaiseBb) &&
      typeof state.raiseReopened === 'boolean' &&
      Array.isArray(state.actionHistory)
  );
}

export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON request' }, { status: 400 });
  }
  if (!body || typeof body !== 'object') {
    return NextResponse.json({ error: 'Invalid resolver request' }, { status: 400 });
  }
  const input = body as {
    modelVersion?: unknown;
    depthBb?: unknown;
    stateHash?: unknown;
    state?: unknown;
  };
  if (
    typeof input.modelVersion !== 'string' ||
    typeof input.depthBb !== 'number' ||
    typeof input.stateHash !== 'string' ||
    !/^[a-f0-9]{64}$/.test(input.stateHash) ||
    !isQueryState(input.state)
  ) {
    return NextResponse.json({ error: 'Invalid resolver request' }, { status: 400 });
  }
  const state = input.state;
  const manifest = modelForFullDepth(input.depthBb);
  const resolver = manifest?.runtime;
  if (
    !manifest ||
    manifest.version !== input.modelVersion ||
    state.modelVersion !== input.modelVersion ||
    state.depthBb !== input.depthBb ||
    resolver?.kind !== 'rust-continual-resolver-v1'
  ) {
    return NextResponse.json(
      { error: 'No pinned continual resolver matches this hand' },
      { status: 404 }
    );
  }
  try {
    const result = await practiceSolverProcess().query(
      resolverQueryPayload(
        state,
        input.stateHash,
        input.modelVersion,
        input.depthBb
      )
    );
    const resolved = result as {
      schema?: unknown;
      modelVersion?: unknown;
      depthBb?: unknown;
      stateHash?: unknown;
      networkSha256?: unknown;
      rangePolicySha256?: unknown;
      valueNetworkSha256?: unknown;
      preflopActionValuesSha256?: unknown;
      actions?: unknown;
      maximumProbabilitySumError?: unknown;
    };
    if (
      resolved.schema !== 'hu-practice-continual-resolver-query-v1' ||
      resolved.modelVersion !== input.modelVersion ||
      resolved.depthBb !== input.depthBb ||
      resolved.stateHash !== input.stateHash ||
      resolved.networkSha256 !== PRACTICE_RESOLVER_IDENTITY.networkSha256 ||
      resolved.rangePolicySha256 !==
        PRACTICE_RESOLVER_IDENTITY.rangePolicySha256 ||
      resolved.valueNetworkSha256 !==
        PRACTICE_RESOLVER_IDENTITY.valueNetworkSha256 ||
      resolved.preflopActionValuesSha256 !==
        PRACTICE_RESOLVER_IDENTITY.preflopActionValuesSha256 ||
      !Array.isArray(resolved.actions) ||
      typeof resolved.maximumProbabilitySumError !== 'number' ||
      resolved.maximumProbabilitySumError > 1e-6
    ) {
      throw new Error('The resolver response does not match its pinned manifest');
    }
    return NextResponse.json(resolved, {
      headers: { 'Cache-Control': 'private, no-store' },
    });
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : 'The pinned practice resolver is unavailable',
      },
      {
        status: 503,
        headers: { 'Cache-Control': 'private, no-store' },
      }
    );
  }
}
