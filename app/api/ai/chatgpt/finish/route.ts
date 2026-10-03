import { NextRequest, NextResponse } from 'next/server';
import { finishSignIn, ATTEMPT_COOKIE, SESSION_COOKIE } from '@/lib/server/chatgpt-auth';

export const runtime = 'nodejs';

export async function GET(req: NextRequest) {
  try {
    const id = await finishSignIn(req);
    const response = NextResponse.redirect(new URL('/solver?chatgpt=connected', req.url));
    response.cookies.set(SESSION_COOKIE, id, { httpOnly: true, sameSite: 'lax', path: '/', maxAge: 30 * 86400 });
    response.cookies.set(ATTEMPT_COOKIE, '', { path: '/api/ai/chatgpt', maxAge: 0 });
    response.headers.set('Cache-Control', 'no-store');
    response.headers.set('Referrer-Policy', 'no-referrer');
    return response;
  } catch {
    const response = NextResponse.redirect(new URL('/solver?chatgpt=failed', req.url));
    response.cookies.set(ATTEMPT_COOKIE, '', { path: '/api/ai/chatgpt', maxAge: 0 });
    return response;
  }
}
