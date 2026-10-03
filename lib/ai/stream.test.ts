import { describe, expect, it } from 'vitest';
import { normalizeStream } from './stream';

function response(events: object[]) {
  return new Response(events.map((event) => 'data: ' + JSON.stringify(event) + '\r\n\r\n').join('')).body!;
}

describe('ChatGPT Responses streaming', () => {
  it('forwards text deltas and completes on the terminal event', async () => {
    const stream = normalizeStream(response([
      { type: 'response.output_text.delta', delta: 'Real ' },
      { type: 'response.output_text.delta', delta: 'answer' },
      { type: 'response.completed' },
    ]), 'responses');
    expect(await new Response(stream).text()).toBe('Real answer');
  });
  it('surfaces a usage error after partial text', async () => {
    const stream = normalizeStream(response([
      { type: 'response.output_text.delta', delta: 'Partial' },
      { type: 'response.failed', response: { error: { message: 'Plan usage limit reached' } } },
    ]), 'responses');
    expect(await new Response(stream).text()).toContain('Plan usage limit reached');
  });
  it('does not treat an interrupted stream as a completed response', async () => {
    const stream = normalizeStream(response([{ type: 'response.output_text.delta', delta: 'Partial' }]), 'responses');
    expect(await new Response(stream).text()).toContain('ended before the response completed');
  });
});
