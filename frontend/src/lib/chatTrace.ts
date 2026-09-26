export type TraceStepKind = "status" | "round" | "tool";

export type TraceStep = {
  id: number;
  kind: TraceStepKind;
  label: string;
  status: "running" | "done" | "error";
  startedAt: number;
  finishedAt?: number;
  detail?: string;
};

export function reduceTracePushStep(
  prev: TraceStep[],
  step: Omit<TraceStep, "id" | "startedAt"> & { startedAt?: number },
  id: number,
  now = Date.now(),
): TraceStep[] {
  const finished = prev.map((s) =>
    s.status === "running" ? { ...s, status: "done" as const, finishedAt: now } : s,
  );
  return [...finished, { ...step, id, startedAt: step.startedAt ?? now } as TraceStep];
}

export function reduceTraceFinishStep(
  prev: TraceStep[],
  stepId: number,
  patch?: Partial<TraceStep>,
  now = Date.now(),
): TraceStep[] {
  return prev.map((s) =>
    s.id === stepId ? { ...s, status: "done" as const, finishedAt: now, ...patch } : s,
  );
}

export function reduceTraceFinishAllRunning(prev: TraceStep[], now = Date.now()): TraceStep[] {
  return prev.map((s) =>
    s.status === "running" ? { ...s, status: "done" as const, finishedAt: now } : s,
  );
}
