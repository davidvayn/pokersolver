import { NextRequest, NextResponse } from 'next/server';
import { receiveCallback } from '@/lib/server/chatgpt-auth';

export const runtime = 'nodejs';

export async function GET(req: NextRequest) {
  try {
    return NextResponse.redirect(receiveCallback(req));
  } catch {
    return new NextResponse('ChatGPT sign-in expired or could not be verified. Return to Poker Lab and try again.', {
      status: 400, headers: { 'Content-Type': 'text/plain', 'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer' },
    });
  }
}
