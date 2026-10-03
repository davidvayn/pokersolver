import { NextRequest, NextResponse } from 'next/server';
import {
  accountModels, assertLocalRequest, beginSignIn, getSession, signOut,
  ATTEMPT_COOKIE, SESSION_COOKIE,
  type ChatGptModel,
} from '@/lib/server/chatgpt-auth';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(req: NextRequest) {
  try {
    assertLocalRequest(req);
    const session = getSession(req);
    if (!session) return NextResponse.json({ available: true, connected: false }, { headers: { 'Cache-Control': 'no-store' } });
    let models: ChatGptModel[] = [];
    let error = '';
    try { models = await accountModels(session); }
    catch (caught) { error = (caught as Error).message; }
    return NextResponse.json({
      available: true, connected: true, registration: session.registration, models, error,
    }, { headers: { 'Cache-Control': 'no-store' } });
  } catch {
    return NextResponse.json({ available: false, connected: false }, { headers: { 'Cache-Control': 'no-store' } });
  }
}

export async function POST(req: NextRequest) {
  try {
    assertLocalRequest(req, true);
    const body = await req.json();
    if (body.action === 'disconnect') {
      const revoked = await signOut(req);
      const response = NextResponse.json({ revoked });
      response.cookies.delete(SESSION_COOKIE);
      return response;
    }
    if (body.action !== 'connect') return NextResponse.json({ error: 'Unknown action.' }, { status: 400 });
    const attempt = beginSignIn(req, body.hostId, body.registration);
    const response = NextResponse.json({ authorizationUrl: attempt.authorizationUrl });
    response.cookies.set(ATTEMPT_COOKIE, attempt.browserId, {
      httpOnly: true, sameSite: 'lax', path: '/api/ai/chatgpt', maxAge: 600,
    });
    return response;
  } catch (caught) {
    return NextResponse.json({ error: (caught as Error).message }, { status: 400 });
  }
}
