import { describe, expect, it } from "vitest";

import { reduceTraceFinishAllRunning, reduceTracePushStep } from "./chatTrace";

// r3_item08
describe("r3_item08_trace_reducer_unit", () => {
  it("finishes running steps when the stream completes in one chunk", () => {
    let steps = reduceTracePushStep(
      [],
      { kind: "status", label: "Calling model…", status: "done" },
      1,
      1000,
    );
    steps = reduceTracePushStep(
      steps,
      { kind: "tool", label: "spotify_me", status: "running" },
      2,
      1001,
    );
    steps = reduceTraceFinishAllRunning(steps, 1002);
    const tool = steps.find((s) => s.label === "spotify_me");
    expect(tool?.status).toBe("done");
    expect(tool?.finishedAt).toBe(1002);
  });
});
