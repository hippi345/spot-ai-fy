import {
  reduceTraceFinishAllRunning,
  reduceTraceFinishStep,
  reduceTracePushStep,
  type TraceStep,
} from "./chatTrace";

export type SseEventHandler = (event: Record<string, unknown>) => void;

/** Parse one SSE block (lines separated by blank line) into JSON events. */
export function parseSseBlock(block: string): Record<string, unknown>[] {
  const events: Record<string, unknown>[] = [];
  for (const rawLine of block.split("\n")) {
    const line = rawLine.trim();
    if (!line.startsWith("data:")) continue;
    const payload = line.slice(5).trim();
    if (!payload) continue;
    try {
      events.push(JSON.parse(payload) as Record<string, unknown>);
    } catch (err) {
      if (err instanceof SyntaxError) continue;
      throw err;
    }
  }
  return events;
}

/** Feed a ReadableStream of SSE bytes through the same parsing path as the chat UI. */
export async function consumeSseReadableStream(
  stream: ReadableStream<Uint8Array>,
  onEvent: SseEventHandler,
): Promise<void> {
  const reader = stream.getReader();
  const dec = new TextDecoder();
  let buf = "";
  const drainBuf = () => {
    for (;;) {
      const sep = buf.indexOf("\n\n");
      if (sep === -1) break;
      const block = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      for (const ev of parseSseBlock(block)) {
        onEvent(ev);
      }
    }
  };
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    drainBuf();
  }
  buf += dec.decode();
  drainBuf();
  const tail = buf.trim();
  if (tail.startsWith("data:")) {
    for (const ev of parseSseBlock(tail)) {
      onEvent(ev);
    }
  }
}

/** Apply chat SSE events to the trace reducer (subset of App.tsx handleEvent). */
export function applyChatSseEventToTrace(
  prev: TraceStep[],
  event: Record<string, unknown>,
  nextId: () => number,
  now = Date.now(),
): TraceStep[] {
  const typ = String(event.type || "");
  let steps = prev;
  switch (typ) {
    case "tool_start": {
      const name = String(event.name ?? "tool");
      steps = reduceTracePushStep(
        steps,
        { kind: "tool", label: name, status: "running" },
        nextId(),
        now,
      );
      break;
    }
    case "tool_done": {
      steps = reduceTraceFinishAllRunning(steps, now);
      break;
    }
    case "final":
      steps = reduceTraceFinishAllRunning(steps, now);
      break;
    case "status": {
      const msg = String(event.message ?? "");
      if (msg.trim()) {
        const id = nextId();
        steps = reduceTracePushStep(
          steps,
          { kind: "status", label: msg, status: "done" },
          id,
          now,
        );
        steps = reduceTraceFinishStep(steps, id, undefined, now);
      }
      break;
    }
    default:
      break;
  }
  return steps;
}
