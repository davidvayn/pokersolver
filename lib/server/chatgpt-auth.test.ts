import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { NextRequest } from 'next/server';
import { exportJWK, generateKeyPair, SignJWT, type JWK } from 'jose';
import {
  accessToken, assertLocalRequest, beginSignIn, finishSignIn, getSession,
  receiveCallback, ATTEMPT_COOKIE, SESSION_COOKIE,
} from './chatgpt-auth';

vi.mock('server-only', () => ({}));
const keys = vi.hoisted(() => ({ keys: [] as JWK[] }));
vi.mock('jose', async (original) => {
  const actual = await original<typeof import('jose')>();
  return {
    ...actual,
    createRemoteJWKSet: () => (...args: Parameters<ReturnType<typeof actual.createLocalJWKSet>>) =>
      actual.createLocalJWKSet(keys)(...args),
  };
});

const HOST = 'urn:uuid:00000000-0000-4000-8000-000000000000';
const ORIGIN = 'http://localhost:3000';
const CLIENT = 'oaiapp_test';
let key: CryptoKey;

beforeAll(async () => {
  const pair = await generateKeyPair('RS256');
  key = pair.privateKey;
  keys.keys = [{ ...await exportJWK(pair.publicKey), kid: 'test', alg: 'RS256' }];
});
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });

function start(registration?: { clientId: string; subject: string }) {
  const result = beginSignIn(new NextRequest(ORIGIN + '/api/ai/chatgpt', {
    method: 'POST', headers: { Origin: ORIGIN },
  }), HOST, registration);
  return { ...result, params: new URL(result.authorizationUrl).searchParams };
}
function callback(params: URLSearchParams, clientId = CLIENT) {
  const uri = new URL(params.get('redirect_uri')!);
  uri.search = new URLSearchParams({
    state: params.get('state')!, code: 'single-use-code', client_id: clientId,
  }).toString();
  return new NextRequest(uri);
}
function finishRequest(attempt: ReturnType<typeof start>) {
  const finish = receiveCallback(callback(attempt.params));
  return new NextRequest(finish, {
    headers: { Cookie: `${ATTEMPT_COOKIE}=${attempt.browserId}` },
  });
}
async function tokens(nonce: string, claims: Record<string, unknown> = {}) {
  const idToken = await new SignJWT({ sub: 'test-account', nonce, ...claims })
    .setProtectedHeader({ alg: 'RS256', kid: 'test' })
    .setIssuer('https://auth.openai.com').setAudience(CLIENT)
    .setIssuedAt().setExpirationTime('1h').sign(key);
  return {
    access_token: 'access-token', refresh_token: 'refresh-token', id_token: idToken,
    scope: 'openid chatgpt.tokens.use.direct offline_access', expires_in: 3600,
  };
}

describe('local ChatGPT authorization', () => {
  it('uses fresh PKCE and state, stable host identity, and the loopback callback', () => {
    const first = start();
    const second = start();
    expect(first.params.get('client_id')).toBe('dynamic_agent_client');
    expect(first.params.get('agent_name_hint')).toBe('Poker Lab');
    expect(first.params.get('ext_agent_host_id')).toBe(HOST);
    expect(first.params.get('redirect_uri')).toBe('http://127.0.0.1:3000/api/ai/chatgpt/callback');
    expect(first.params.get('state')).not.toBe(second.params.get('state'));
    expect(first.params.get('code_challenge')).not.toBe(second.params.get('code_challenge'));
    expect(first.params.get('code_challenge_method')).toBe('S256');
  });

  it('rejects hosted requests and cross-origin mutations', () => {
    expect(() => assertLocalRequest(new NextRequest('https://example.com/api/ai/chatgpt'))).toThrow('local app only');
    expect(() => assertLocalRequest(new NextRequest(ORIGIN + '/api/ai/chatgpt', {
      method: 'POST', headers: { Origin: 'https://example.com' },
    }), true)).toThrow('must come from this app');
  });

  it('requires the originating browser cookie before exchanging a code', async () => {
    const attempt = start();
    const finish = receiveCallback(callback(attempt.params));
    const upstream = vi.fn();
    vi.stubGlobal('fetch', upstream);
    await expect(finishSignIn(new NextRequest(finish))).rejects.toThrow('could not be verified');
    expect(upstream).not.toHaveBeenCalled();
  });

  it('verifies identity and uses the issued client ID and original PKCE redirect for exchange', async () => {
    const attempt = start();
    const tokenSet = await tokens(attempt.params.get('nonce')!);
    const upstream = vi.fn().mockResolvedValue(Response.json(tokenSet));
    vi.stubGlobal('fetch', upstream);
    const request = finishRequest(attempt);
    const sessionId = await finishSignIn(request);
    const session = getSession(new NextRequest(ORIGIN + '/api/ai/chatgpt', {
      headers: { Cookie: `${SESSION_COOKIE}=${sessionId}` },
    }));
    expect(session?.registration).toEqual({ clientId: CLIENT, subject: 'test-account' });
    const form = upstream.mock.calls[0][1].body as URLSearchParams;
    expect(form.get('client_id')).toBe(CLIENT);
    expect(form.get('redirect_uri')).toBe(attempt.params.get('redirect_uri'));
    const challenge = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(form.get('code_verifier')!));
    expect(Buffer.from(challenge).toString('base64url')).toBe(attempt.params.get('code_challenge'));
    await expect(finishSignIn(request)).rejects.toThrow('could not be verified');
  });

  it('rejects an expired transaction and a changed issued client ID', async () => {
    const attempt = start({ clientId: CLIENT, subject: 'test-account' });
    expect(() => receiveCallback(callback(attempt.params, 'oaiapp_other'))).toThrow('expected app registration');
    const expired = start();
    const request = finishRequest(expired);
    vi.useFakeTimers();
    vi.setSystemTime(Date.now() + 11 * 60_000);
    await expect(finishSignIn(request)).rejects.toThrow('could not be verified');
  });

  it('rejects a signed token with a different nonce or account', async () => {
    for (const claims of [{ nonce: 'wrong' }, { sub: 'another-account' }]) {
      const attempt = start({ clientId: CLIENT, subject: 'test-account' });
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(Response.json(await tokens(attempt.params.get('nonce')!, claims))));
      await expect(finishSignIn(finishRequest(attempt))).rejects.toThrow('different account');
    }
  });

  it('rejects an untrusted token signature and a grant without plan permission', async () => {
    const attempt = start();
    const tokenSet = await tokens(attempt.params.get('nonce')!);
    const parts = tokenSet.id_token.split('.');
    parts[2] = parts[2][0] === 'A' ? 'B' + parts[2].slice(1) : 'A' + parts[2].slice(1);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(Response.json({ ...tokenSet, id_token: parts.join('.') })));
    await expect(finishSignIn(finishRequest(attempt))).rejects.toThrow();
    const noPermission = start();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(Response.json({ ...tokenSet, scope: 'openid email' })));
    await expect(finishSignIn(finishRequest(noPermission))).rejects.toThrow('plan access was not granted');
  });

  it('serializes concurrent refreshes and retains the rotated credential', async () => {
    const attempt = start();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(Response.json(await tokens(attempt.params.get('nonce')!))));
    const id = await finishSignIn(finishRequest(attempt));
    const session = getSession(new NextRequest(ORIGIN + '/api/ai/chatgpt', {
      headers: { Cookie: `${SESSION_COOKIE}=${id}` },
    }))!;
    session.expiresAt = 0;
    const refreshFetch = vi.fn().mockResolvedValue(Response.json({
      access_token: 'new-access', refresh_token: 'rotated-refresh', expires_in: 3600,
    }));
    vi.stubGlobal('fetch', refreshFetch);
    expect(await Promise.all([accessToken(session), accessToken(session)])).toEqual(['new-access', 'new-access']);
    expect(refreshFetch).toHaveBeenCalledTimes(1);
    expect(session.tokens.refresh_token).toBe('rotated-refresh');
  });
});
