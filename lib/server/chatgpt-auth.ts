import 'server-only';

import { createHash, randomBytes } from 'node:crypto';
import { createRemoteJWKSet, jwtVerify } from 'jose';
import type { NextRequest } from 'next/server';

const ISSUER = 'https://auth.openai.com';
const RESOURCE = 'https://api.openai.com/v1';
const DYNAMIC_CLIENT = 'dynamic_agent_client';
const jwks = createRemoteJWKSet(new URL(`${ISSUER}/.well-known/jwks.json`));
export const ATTEMPT_COOKIE = 'poker-chatgpt-attempt';
export const SESSION_COOKIE = 'poker-chatgpt-session';

interface Registration {
  clientId: string;
  subject: string;
  email?: string;
}
interface Attempt {
  browserId: string;
  origin: string;
  hostId: string;
  state: string;
  nonce: string;
  verifier: string;
  redirectUri: string;
  registration?: Registration;
  expiresAt: number;
  callback?: { code: string; clientId: string };
}
interface TokenSet {
  access_token: string;
  refresh_token: string;
  id_token: string;
  scope: string;
  expires_in: number;
}
interface Session {
  registration: Registration;
  hostId: string;
  tokens: TokenSet;
  expiresAt: number;
  refresh?: Promise<void>;
  models?: { items: ChatGptModel[]; expiresAt: number };
}
export interface ChatGptModel {
  id: string;
  label: string;
}

// Credentials live only in this local server process. Nothing is written to
// disk or browser storage. Restarting the server requires signing in again.
const localRuntime = globalThis as typeof globalThis & {
  pokerChatGpt?: {
    attempts: Map<string, Attempt>;
    sessions: Map<string, Session>;
  };
};
const store = localRuntime.pokerChatGpt ??= {
  attempts: new Map(),
  sessions: new Map(),
};
const random = () => randomBytes(32).toString('base64url');

export function assertLocalRequest(req: NextRequest, mutation = false) {
  const url = new URL(req.url);
  if (
    url.protocol !== 'http:' ||
    !['localhost', '127.0.0.1'].includes(url.hostname)
  ) {
    throw new Error('ChatGPT sign-in is currently available in the local app only.');
  }
  if (mutation && req.headers.get('origin') !== url.origin) {
    throw new Error('The request must come from this app.');
  }
}

function pruneAttempts() {
  for (const [state, attempt] of store.attempts) {
    if (attempt.expiresAt < Date.now()) store.attempts.delete(state);
  }
}

export function beginSignIn(
  req: NextRequest,
  hostId: string,
  registration?: Registration,
) {
  assertLocalRequest(req, true);
  if (!/^urn:uuid:[0-9a-f-]{36}$/i.test(hostId)) {
    throw new Error('Invalid local host identifier.');
  }
  if (registration && (
    !/^oaiapp_[a-zA-Z0-9_-]+$/.test(registration.clientId) ||
    !registration.subject
  )) {
    throw new Error('Invalid saved ChatGPT registration.');
  }
  pruneAttempts();
  const origin = new URL(req.url).origin;
  const callback = new URL('/api/ai/chatgpt/callback', origin);
  callback.hostname = '127.0.0.1';
  const attempt: Attempt = {
    browserId: random(), origin, hostId, registration,
    state: random(), nonce: random(), verifier: random(),
    redirectUri: callback.href, expiresAt: Date.now() + 10 * 60_000,
  };
  store.attempts.set(attempt.state, attempt);
  const authorization = new URL(`${ISSUER}/api/accounts/authorize`);
  authorization.search = new URLSearchParams({
    client_id: registration?.clientId ?? DYNAMIC_CLIENT,
    ...(registration ? {} : { agent_name_hint: 'Poker Lab' }),
    ext_agent_host_id: hostId,
    response_type: 'code',
    redirect_uri: attempt.redirectUri,
    scope: 'openid profile email offline_access resource.invoke chatgpt.tokens.use.direct',
    resource: RESOURCE,
    state: attempt.state,
    nonce: attempt.nonce,
    code_challenge_method: 'S256',
    code_challenge: createHash('sha256').update(attempt.verifier).digest('base64url'),
  }).toString();
  return { authorizationUrl: authorization.href, browserId: attempt.browserId };
}

// The provider returns to 127.0.0.1. Bridge back to the original localhost
// origin to verify its HttpOnly attempt cookie before exchanging the code.
export function receiveCallback(req: NextRequest): string {
  assertLocalRequest(req);
  pruneAttempts();
  const params = new URL(req.url).searchParams;
  const state = params.get('state') ?? '';
  const attempt = store.attempts.get(state);
  if (!attempt || attempt.callback) throw new Error('This sign-in attempt expired or was already used.');
  if (params.has('error')) {
    store.attempts.delete(state);
    return new URL('/solver?chatgpt=denied', attempt.origin).href;
  }
  const code = params.get('code');
  const clientId = params.get('client_id') ?? attempt.registration?.clientId;
  if (!code || !clientId || clientId === DYNAMIC_CLIENT ||
      !/^oaiapp_[a-zA-Z0-9_-]+$/.test(clientId) ||
      (attempt.registration && clientId !== attempt.registration.clientId)) {
    store.attempts.delete(state);
    throw new Error('ChatGPT did not return the expected app registration.');
  }
  attempt.callback = { code, clientId };
  const finish = new URL('/api/ai/chatgpt/finish', attempt.origin);
  finish.searchParams.set('state', state);
  return finish.href;
}

async function readTokens(response: Response, previous?: TokenSet): Promise<TokenSet> {
  if (!response.ok) throw new Error('ChatGPT authorization failed. Please sign in again.');
  const data = await response.json() as Partial<TokenSet>;
  const tokens = {
    ...data,
    id_token: data.id_token ?? previous?.id_token,
    scope: data.scope ?? previous?.scope,
  } as TokenSet;
  if (!tokens.access_token || !tokens.refresh_token || !tokens.id_token ||
      !Number.isFinite(tokens.expires_in) || tokens.expires_in <= 0 ||
      !tokens.scope?.split(' ').includes('chatgpt.tokens.use.direct')) {
    throw new Error('ChatGPT plan access was not granted. Sign in and allow plan usage.');
  }
  return tokens;
}

export async function finishSignIn(req: NextRequest) {
  assertLocalRequest(req);
  const state = new URL(req.url).searchParams.get('state') ?? '';
  const attempt = store.attempts.get(state);
  if (!attempt || attempt.expiresAt < Date.now() ||
      attempt.origin !== new URL(req.url).origin ||
      req.cookies.get(ATTEMPT_COOKIE)?.value !== attempt.browserId) {
    throw new Error('This sign-in attempt could not be verified. Please start again.');
  }
  store.attempts.delete(state); // One-time consumption, including failed exchanges.
  if (!attempt.callback) throw new Error('ChatGPT has not completed authorization.');
  const { code, clientId } = attempt.callback;
  const tokens = await readTokens(await fetch(`${ISSUER}/api/accounts/oauth/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'authorization_code', client_id: clientId, code,
      code_verifier: attempt.verifier, redirect_uri: attempt.redirectUri,
      resource: RESOURCE,
    }),
    cache: 'no-store',
  }));
  const { payload } = await jwtVerify(tokens.id_token, jwks, {
    issuer: ISSUER, audience: clientId, algorithms: ['RS256'],
    requiredClaims: ['sub', 'exp', 'iat'], clockTolerance: 5,
  });
  if (payload.nonce !== attempt.nonce || !payload.sub ||
      (attempt.registration && payload.sub !== attempt.registration.subject)) {
    throw new Error('ChatGPT returned a different account. Please start a new sign-in.');
  }
  const sessionId = random();
  const previousId = req.cookies.get(SESSION_COOKIE)?.value;
  if (previousId) await endSession(previousId);
  store.sessions.set(sessionId, {
    registration: {
      clientId, subject: payload.sub,
      ...(typeof payload.email === 'string' ? { email: payload.email } : {}),
    },
    hostId: attempt.hostId, tokens,
    expiresAt: Date.now() + tokens.expires_in * 1000,
  });
  return sessionId;
}

export function getSession(req: NextRequest): Session | undefined {
  assertLocalRequest(req);
  const id = req.cookies.get(SESSION_COOKIE)?.value;
  return id ? store.sessions.get(id) : undefined;
}

export async function accessToken(session: Session) {
  if (session.expiresAt > Date.now() + 60_000) return session.tokens.access_token;
  if (!session.refresh) {
    session.refresh = (async () => {
      const tokens = await readTokens(await fetch(`${ISSUER}/api/accounts/oauth/token`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
          grant_type: 'refresh_token',
          client_id: session.registration.clientId,
          refresh_token: session.tokens.refresh_token,
          resource: RESOURCE,
        }),
        cache: 'no-store',
      }), session.tokens);
      if (tokens.id_token !== session.tokens.id_token) {
        const { payload } = await jwtVerify(tokens.id_token, jwks, {
          issuer: ISSUER, audience: session.registration.clientId,
          algorithms: ['RS256'], requiredClaims: ['sub', 'exp', 'iat'], clockTolerance: 5,
        });
        if (payload.sub !== session.registration.subject) throw new Error('ChatGPT account changed during refresh. Please sign in again.');
      }
      session.tokens = tokens;
      session.expiresAt = Date.now() + tokens.expires_in * 1000;
    })().finally(() => { session.refresh = undefined; });
  }
  await session.refresh;
  return session.tokens.access_token;
}

export async function accountModels(session: Session): Promise<ChatGptModel[]> {
  if (session.models && session.models.expiresAt > Date.now()) return session.models.items;
  const response = await fetch('https://api.openai.com/v1/models', {
    headers: { Authorization: `Bearer ${await accessToken(session)}` },
    cache: 'no-store',
  });
  if (!response.ok) throw new Error('Could not load your ChatGPT models. Please try again.');
  const data = await response.json() as {
    models?: { slug: string; display_name: string; visibility: string }[];
  };
  const items = (data.models ?? [])
    .filter((model) => model.visibility === 'list' && model.slug)
    .map((model) => ({ id: model.slug, label: model.display_name || model.slug }));
  session.models = { items, expiresAt: Date.now() + 5 * 60_000 };
  return items;
}

async function endSession(id: string): Promise<boolean> {
  const session = store.sessions.get(id);
  if (!session) return true;
  if (session.refresh) await session.refresh.catch(() => {});
  let revoked = false;
  try {
    for (let attempt = 0; attempt < 3; attempt++) {
      const response = await fetch(`${ISSUER}/api/accounts/oauth/revoke`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
          token: session.tokens.refresh_token, token_type_hint: 'refresh_token',
          client_id: session.registration.clientId,
        }),
        cache: 'no-store',
      });
      if (response.ok) { revoked = true; break; }
      if (response.status < 500) break;
      await new Promise((resolve) => setTimeout(resolve, 250 * (attempt + 1)));
    }
  } catch { /* Clear local access even when the provider is unreachable. */ }
  store.sessions.delete(id);
  return revoked;
}

export async function signOut(req: NextRequest) {
  assertLocalRequest(req, true);
  const id = req.cookies.get(SESSION_COOKIE)?.value;
  return id ? endSession(id) : true;
}
