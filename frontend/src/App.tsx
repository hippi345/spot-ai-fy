import { useCallback, useEffect, useMemo, useRef, useState } from "react";



type Session = {
  signed_in: boolean;
  device_id: string | null;
  spotify_granted_scopes?: string | null;
  spotify_playlist_write_ok?: boolean | null;
};



type SpotifyDevice = {

  id: string;

  name: string;

  is_active: boolean;

  type: string;

};



type ChatMessage = { role: "user" | "assistant"; text: string };



type TraceStepKind = "status" | "round" | "tool";

type TraceStep = {
  id: number;
  kind: TraceStepKind;
  label: string;
  detail?: string;
  startedAt: number;
  finishedAt?: number;
  status: "running" | "done" | "error";
};



type LlmStatus = {

  provider?: string;

  env_provider?: string;

  ui_override?: boolean;

  configured_host?: string;

  configured_model: string;

  env_ollama_model?: string;

  ollama_model_ui_override?: boolean;

  env_gemini_model?: string;

  gemini_model_ui_override?: boolean;

  reachable: boolean;

  models: string[] | null;

  model_installed?: boolean;

  error: string | null;

};



async function readJson<T>(res: Response): Promise<T> {

  const text = await res.text();

  if (!res.ok) {

    let msg = text || res.statusText;

    try {

      const j = JSON.parse(text) as { detail?: string };

      if (typeof j.detail === "string") msg = j.detail;

    } catch {

      /* keep msg */

    }

    throw new Error(msg);

  }

  return (text ? (JSON.parse(text) as T) : ({} as T));

}



export function App() {

  const [session, setSession] = useState<Session | null>(null);

  const [devices, setDevices] = useState<SpotifyDevice[]>([]);

  const [deviceId, setDeviceId] = useState("");

  const [loadingDevices, setLoadingDevices] = useState(false);

  const [banner, setBanner] = useState<string | null>(null);

  const [error, setError] = useState<string | null>(null);



  const [input, setInput] = useState("");

  const [sending, setSending] = useState(false);

  const [messages, setMessages] = useState<ChatMessage[]>([]);

  const [traceSteps, setTraceSteps] = useState<TraceStep[]>([]);

  const [showTraceDetail, setShowTraceDetail] = useState<boolean>(() => {
    try {
      return localStorage.getItem("spotaify.showTraceDetail") === "1";
    } catch {
      return false;
    }
  });

  const [nowTick, setNowTick] = useState<number>(() => Date.now());

  const [liveReply, setLiveReply] = useState("");

  // Auto-scroll the chat-messages container to the bottom when a new message
  // arrives or the streaming reply grows — but only if the user was already
  // near the bottom, so we don't yank them away from older messages they
  // scrolled up to read.
  const messagesRef = useRef<HTMLDivElement | null>(null);
  const userPinnedToBottomRef = useRef<boolean>(true);

  const handleMessagesScroll = useCallback(() => {
    const el = messagesRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    userPinnedToBottomRef.current = distanceFromBottom < 80;
  }, []);

  useEffect(() => {
    if (!userPinnedToBottomRef.current) return;
    const el = messagesRef.current;
    if (!el) return;
    // Two-step scroll: update synchronously, then again after layout settles
    // (lets late-rendering content like the streaming bubble grow before we
    // commit the final scroll position).
    el.scrollTop = el.scrollHeight;
    const id = window.requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight;
    });
    return () => window.cancelAnimationFrame(id);
  }, [messages.length, liveReply, sending, traceSteps.length]);

  const [llm, setLlm] = useState<LlmStatus | null>(null);

  const [llmPick, setLlmPick] = useState<"ollama" | "gemini">("ollama");

  const [llmSaving, setLlmSaving] = useState(false);

  const [ollamaModelSelect, setOllamaModelSelect] = useState("__env__");

  const [ollamaCustomModel, setOllamaCustomModel] = useState("");

  const [geminiModelSelect, setGeminiModelSelect] = useState("__env__");

  const [geminiCustomModel, setGeminiCustomModel] = useState("");



  const refreshLlm = useCallback(async () => {

    try {

      const data = await readJson<LlmStatus>(await fetch("/api/llm"));

      setLlm(data);

      const p = data.provider === "gemini" ? "gemini" : "ollama";

      setLlmPick(p);

      const eff = (data.configured_model || "").trim();

      if (p === "ollama") {

        const available = data.models ?? [];

        const fromEnv = !data.ollama_model_ui_override;

        if (fromEnv) {

          setOllamaModelSelect("__env__");

          setOllamaCustomModel(eff || (data.env_ollama_model ?? "").trim());

        } else if (available.includes(eff)) {

          setOllamaModelSelect(eff);

          setOllamaCustomModel(eff);

        } else {

          setOllamaModelSelect("__custom__");

          setOllamaCustomModel(eff);

        }

      } else {

        const available = data.models ?? [];

        const fromEnv = !data.gemini_model_ui_override;

        if (fromEnv) {

          setGeminiModelSelect("__env__");

          setGeminiCustomModel(eff || (data.env_gemini_model ?? "").trim());

        } else if (available.includes(eff)) {

          setGeminiModelSelect(eff);

          setGeminiCustomModel(eff);

        } else {

          setGeminiModelSelect("__custom__");

          setGeminiCustomModel(eff);

        }

      }

    } catch {

      setLlm({

        provider: "ollama",

        env_provider: "ollama",

        ui_override: false,

        configured_host: "",

        configured_model: "",

        reachable: false,

        models: null,

        error: "Could not load /api/llm (is the API running?)",

      });

    }

  }, []);



  const applyLlmProvider = async () => {

    setLlmSaving(true);

    setError(null);

    try {

      await readJson(

        await fetch("/api/llm/provider", {

          method: "POST",

          headers: { "Content-Type": "application/json" },

          body: JSON.stringify({ provider: llmPick }),

        }),

      );

      await refreshLlm();

      setBanner(`Backend: ${llmPick}.`);

    } catch (e) {

      setError(e instanceof Error ? e.message : "Could not save LLM choice");

    } finally {

      setLlmSaving(false);

    }

  };



  const resetLlmProvider = async () => {

    setLlmSaving(true);

    setError(null);

    try {

      await readJson(await fetch("/api/llm/provider", { method: "DELETE" }));

      await refreshLlm();

      setBanner("Using LLM_PROVIDER from backend/.env.");

    } catch (e) {

      setError(e instanceof Error ? e.message : "Could not reset LLM choice");

    } finally {

      setLlmSaving(false);

    }

  };



  const applyOllamaModel = async () => {

    if (!llm || llm.provider !== "ollama") return;

    setLlmSaving(true);

    setError(null);

    try {

      if (ollamaModelSelect === "__env__") {

        await readJson(await fetch("/api/llm/ollama-model", { method: "DELETE" }));

        setBanner("Ollama model from .env again.");

      } else {

        const tag =

          ollamaModelSelect === "__custom__" ? ollamaCustomModel.trim() : ollamaModelSelect.trim();

        if (!tag) throw new Error("Enter an Ollama model tag (e.g. llama3.1:8b).");

        await readJson(

          await fetch("/api/llm/ollama-model", {

            method: "POST",

            headers: { "Content-Type": "application/json" },

            body: JSON.stringify({ model: tag }),

          }),

        );

        setBanner(`Ollama: ${tag}.`);

      }

      await refreshLlm();

    } catch (e) {

      setError(e instanceof Error ? e.message : "Could not save Ollama model");

    } finally {

      setLlmSaving(false);

    }

  };



  const ollamaModelApplyDisabled = useMemo(() => {

    if (!llm || llm.provider !== "ollama") return true;

    if (ollamaModelSelect === "__env__") return !llm.ollama_model_ui_override;

    if (ollamaModelSelect === "__custom__") {

      const t = ollamaCustomModel.trim();

      if (!t) return true;

      return Boolean(llm.ollama_model_ui_override && t === (llm.configured_model || "").trim());

    }

    return Boolean(llm.ollama_model_ui_override && ollamaModelSelect === (llm.configured_model || "").trim());

  }, [llm, ollamaModelSelect, ollamaCustomModel]);



  const applyGeminiModel = async () => {

    if (!llm || llm.provider !== "gemini") return;

    setLlmSaving(true);

    setError(null);

    try {

      if (geminiModelSelect === "__env__") {

        await readJson(await fetch("/api/llm/gemini-model", { method: "DELETE" }));

        setBanner("Gemini model from .env again.");

      } else {

        const tag =

          geminiModelSelect === "__custom__" ? geminiCustomModel.trim() : geminiModelSelect.trim();

        if (!tag) throw new Error("Enter a Gemini model name (e.g. gemini-2.5-flash).");

        await readJson(

          await fetch("/api/llm/gemini-model", {

            method: "POST",

            headers: { "Content-Type": "application/json" },

            body: JSON.stringify({ model: tag }),

          }),

        );

        setBanner(`Gemini: ${tag}.`);

      }

      await refreshLlm();

    } catch (e) {

      setError(e instanceof Error ? e.message : "Could not save Gemini model");

    } finally {

      setLlmSaving(false);

    }

  };



  const geminiModelApplyDisabled = useMemo(() => {

    if (!llm || llm.provider !== "gemini") return true;

    if (geminiModelSelect === "__env__") return !llm.gemini_model_ui_override;

    if (geminiModelSelect === "__custom__") {

      const t = geminiCustomModel.trim();

      if (!t) return true;

      return Boolean(llm.gemini_model_ui_override && t === (llm.configured_model || "").trim());

    }

    return Boolean(llm.gemini_model_ui_override && geminiModelSelect === (llm.configured_model || "").trim());

  }, [llm, geminiModelSelect, geminiCustomModel]);



  const refreshSession = useCallback(async () => {

    const s = await readJson<Session>(await fetch("/api/session"));

    setSession(s);

    if (s.device_id) setDeviceId(s.device_id);

  }, []);



  const refreshDevices = useCallback(async () => {

    setLoadingDevices(true);

    setError(null);

    try {

      const data = await readJson<{ devices?: SpotifyDevice[] }>(await fetch("/api/devices"));

      setDevices(data.devices ?? []);

    } catch (e) {

      setDevices([]);

      setError(e instanceof Error ? e.message : "Failed to load devices");

    } finally {

      setLoadingDevices(false);

    }

  }, []);



  useEffect(() => {

    const params = new URLSearchParams(window.location.search);

    const spotify = params.get("spotify");

    if (spotify === "connected") {

      setBanner("Spotify connected. Choose a playback device in settings if you have not yet.");

      window.history.replaceState({}, "", window.location.pathname);

    } else if (spotify === "error") {

      const reason = params.get("reason") || "unknown";

      setBanner(`Spotify auth: ${reason}`);

      window.history.replaceState({}, "", window.location.pathname);

    }

    void refreshSession();

    void refreshLlm();

  }, [refreshSession, refreshLlm]);



  useEffect(() => {

    if (session?.signed_in) void refreshDevices();

  }, [session?.signed_in, refreshDevices]);



  useEffect(() => {

    if (!banner) return;

    const id = window.setTimeout(() => setBanner(null), 8000);

    return () => window.clearTimeout(id);

  }, [banner]);



  useEffect(() => {

    if (!error) return;

    const id = window.setTimeout(() => setError(null), 8000);

    return () => window.clearTimeout(id);

  }, [error]);



  useEffect(() => {
    try {
      localStorage.setItem("spotaify.showTraceDetail", showTraceDetail ? "1" : "0");
    } catch {
      /* ignore */
    }
  }, [showTraceDetail]);



  useEffect(() => {
    if (!sending) return;
    const id = window.setInterval(() => setNowTick(Date.now()), 500);
    return () => window.clearInterval(id);
  }, [sending]);



  const deviceOptions = useMemo(() => {

    return devices.map((d) => ({

      value: d.id,

      label: `${d.name} (${d.type})${d.is_active ? " · active" : ""}`,

    }));

  }, [devices]);



  const saveDevice = async () => {

    if (!deviceId) return;

    setError(null);

    await fetch("/api/device", {

      method: "POST",

      headers: { "Content-Type": "application/json" },

      body: JSON.stringify({ device_id: deviceId }),

    });

    await refreshSession();

    setBanner("Playback device saved.");

  };



  const sendChat = async () => {

    const text = input.trim();

    if (!text) return;

    const historyPayload = messages.slice(-40).map((m) => ({ role: m.role, content: m.text }));

    setSending(true);

    setError(null);

    setTraceSteps([]);

    setLiveReply("");

    setInput("");

    setMessages((m) => [...m, { role: "user", text }]);

    const controller = new AbortController();

    const chatTimeoutMs = 900_000;

    const timeoutId = window.setTimeout(() => controller.abort(), chatTimeoutMs);

    let nextStepId = 1;
    const newStepId = () => nextStepId++;

    const pushStep = (step: Omit<TraceStep, "id" | "startedAt"> & { startedAt?: number }) => {
      const id = newStepId();
      const now = Date.now();
      setTraceSteps((prev) => {
        const finished = prev.map((s) =>
          s.status === "running" ? { ...s, status: "done" as const, finishedAt: now } : s,
        );
        return [
          ...finished,
          { ...step, id, startedAt: step.startedAt ?? now } as TraceStep,
        ];
      });
      return id;
    };

    const finishStep = (id: number, patch?: Partial<TraceStep>) => {
      const now = Date.now();
      setTraceSteps((prev) =>
        prev.map((s) =>
          s.id === id
            ? { ...s, status: "done", finishedAt: now, ...patch }
            : s,
        ),
      );
    };

    const finishAllRunning = () => {
      const now = Date.now();
      setTraceSteps((prev) =>
        prev.map((s) =>
          s.status === "running" ? { ...s, status: "done", finishedAt: now } : s,
        ),
      );
    };

    const toolStepIdByName = new Map<string, number>();



    let streamOk = false;

    try {

      const res = await fetch("/api/chat/stream", {

        method: "POST",

        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },

        body: JSON.stringify({ message: text, history: historyPayload }),

        signal: controller.signal,

      });



      if (!res.ok) {

        const errText = await res.text();

        let msg = errText || res.statusText;

        try {

          const j = JSON.parse(errText) as { detail?: string };

          if (typeof j.detail === "string") msg = j.detail;

        } catch {

          /* keep */

        }

        throw new Error(msg);

      }



      const reader = res.body?.getReader();

      if (!reader) throw new Error("No response body");



      const dec = new TextDecoder();

      let buf = "";

      let reply = "";



      const handleEvent = (j: Record<string, unknown>) => {

        const typ = String(j.type || "");

        switch (typ) {

          case "status": {
            const msg = String(j.message ?? "");
            if (msg.trim()) {
              const sid = pushStep({ kind: "status", label: msg, status: "done" });
              finishStep(sid);
            }
            break;
          }

          case "round": {
            const step = String(j.step ?? "?");
            const max = String(j.max ?? "?");
            pushStep({
              kind: "round",
              label: `Round ${step} / ${max} — waiting for model…`,
              status: "running",
            });
            break;
          }

          case "llm_delta":

            reply += String(j.text ?? "");

            setLiveReply(reply);

            break;

          case "tool_start": {
            const name = String(j.name ?? "");
            const id = pushStep({
              kind: "tool",
              label: name || "tool",
              status: "running",
            });
            if (name) toolStepIdByName.set(name, id);
            break;
          }

          case "tool_done": {
            const name = String(j.name ?? "");
            const preview = j.preview ? String(j.preview) : "";
            const id = toolStepIdByName.get(name);
            if (id !== undefined) {
              finishStep(id, { detail: preview || undefined });
              toolStepIdByName.delete(name);
            } else {
              const sid = pushStep({
                kind: "tool",
                label: name || "tool",
                status: "done",
                detail: preview || undefined,
              });
              finishStep(sid, { detail: preview || undefined });
            }
            break;
          }

          case "final":

            reply = String(j.text ?? "");

            setLiveReply(reply);

            finishAllRunning();

            break;

          case "error":

            finishAllRunning();

            throw new Error(String(j.message ?? "Stream error"));

          case "done":

            break;

          default:

            break;

        }

      };



      const drainBuf = () => {

        for (;;) {

          const sep = buf.indexOf("\n\n");

          if (sep === -1) break;

          const block = buf.slice(0, sep);

          buf = buf.slice(sep + 2);

          for (const rawLine of block.split("\n")) {

            const line = rawLine.trim();

            if (!line.startsWith("data:")) continue;

            const payload = line.slice(5).trim();

            if (!payload) continue;

            try {

              handleEvent(JSON.parse(payload) as Record<string, unknown>);

            } catch (err) {

              if (err instanceof SyntaxError) continue;

              throw err;

            }

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

        const payload = tail.slice(5).trim();

        if (payload) {

          try {

            handleEvent(JSON.parse(payload) as Record<string, unknown>);

          } catch (err) {

            if (!(err instanceof SyntaxError)) throw err;

          }

        }

      }



      const emptyish = !reply.trim() || reply.trim() === "No response from model.";

      if (emptyish) {

        setError(

          "The model returned no assistant text and no tool calls (Ollama may stream reasoning without a final answer, or the run was cut short). Try: (1) a shorter, one-step question, (2) another Ollama tag if this one misbehaves with tools, (3) Gemini in Spot-AI-fy, or (4) concrete examples in backend/AGENT_CONTEXT.md."

        );

      } else {

        setMessages((m) => [...m, { role: "assistant", text: reply }]);

        streamOk = true;

      }

    } catch (e) {

      const aborted =

        (e instanceof DOMException && e.name === "AbortError") ||

        (e instanceof Error && e.name === "AbortError");

      setError(

        aborted

          ? `Chat timed out after ${Math.round(chatTimeoutMs / 60_000)} minutes. Try a shorter question, a faster model, or open the UI via the Vite dev server (not file://).`

          : e instanceof Error

            ? e.message

            : "Chat failed"

      );

    } finally {

      window.clearTimeout(timeoutId);

      setSending(false);

      setLiveReply("");

      if (streamOk) setTraceSteps([]);

    }

  };



  const logout = async () => {

    await fetch("/logout", { method: "POST" });

    setSession({
      signed_in: false,
      device_id: null,
      spotify_granted_scopes: null,
      spotify_playlist_write_ok: null,
    });

    setDevices([]);

    setDeviceId("");

    setMessages([]);

    setBanner("Signed out.");

  };



  const formatElapsed = (ms: number): string => {
    if (!Number.isFinite(ms) || ms < 0) return "0.0s";
    if (ms < 10_000) return `${(ms / 1000).toFixed(1)}s`;
    if (ms < 60_000) return `${Math.round(ms / 1000)}s`;
    const m = Math.floor(ms / 60_000);
    const s = Math.round((ms % 60_000) / 1000);
    return `${m}m ${s.toString().padStart(2, "0")}s`;
  };

  const traceIcon = (step: TraceStep): string => {
    if (step.status === "running") return "⟳";
    if (step.kind === "tool") return "✓";
    if (step.kind === "round") return "▸";
    return "•";
  };

  const showTracePanel = llm?.provider === "ollama" && traceSteps.length > 0;

  const statusChips = useMemo(() => {
    const spotifyLabel = `Spotify — ${session?.signed_in ? "Connected" : "Not connected"}`;
    const llmName = llm?.provider === "gemini" ? "Gemini" : "Ollama";
    const llmConnected = Boolean(llm?.reachable);
    const llmLabel = `${llmName} — ${llmConnected ? "Connected" : "Not connected"}`;
    return `${spotifyLabel} · ${llmLabel}`;
  }, [session?.signed_in, llm?.provider, llm?.reachable]);



  return (

    <div className="app">

      <header className="app-header">

        <div className="app-header-main">

          <h1>Spot-AI-fy</h1>

          <p className="tagline">Ask Spotify in plain language — search, playlists, playback.</p>

        </div>

        <p className="status-chips" aria-live="polite">

          {statusChips}

        </p>

      </header>



      {banner ? (

        <div className="banner" role="status">

          <span>{banner}</span>

          <button type="button" className="banner-dismiss" onClick={() => setBanner(null)} aria-label="Dismiss">

            ×

          </button>

        </div>

      ) : null}



      <section className="panel chat-panel" aria-labelledby="chat-heading">

        <div className="chat-panel-head">

          <h2 id="chat-heading" className="panel-title">

            Chat

          </h2>

        </div>

        <div
          className="messages"
          role="log"
          aria-relevant="additions"
          ref={messagesRef}
          onScroll={handleMessagesScroll}
        >

          {messages.map((m, i) => (

            <div key={i} className={`bubble ${m.role}`}>

              {m.text}

            </div>

          ))}

        </div>

        {sending && (liveReply || traceSteps.length > 0) ? (

          <div className="bubble assistant streaming" aria-live="polite">

            {liveReply || "…"}

          </div>

        ) : null}

        {showTracePanel ? (
          <div className="trace-panel" aria-live="polite" aria-label="Agent progress">
            <div className="trace-panel-head">
              <span className="trace-panel-title">
                {sending ? "Agent progress" : "Last run"}
              </span>
              <label className="trace-detail-toggle">
                <input
                  type="checkbox"
                  checked={showTraceDetail}
                  onChange={(e) => setShowTraceDetail(e.target.checked)}
                />
                <span>Show details</span>
              </label>
            </div>
            <ul className="trace-steps">
              {traceSteps.map((step) => {
                const end = step.finishedAt ?? (step.status === "running" ? nowTick : step.startedAt);
                const elapsed = Math.max(0, end - step.startedAt);
                return (
                  <li
                    key={step.id}
                    className={`trace-step trace-step--${step.kind} trace-step--${step.status}`}
                  >
                    <span className="trace-step-icon" aria-hidden="true">
                      {traceIcon(step)}
                    </span>
                    <span className="trace-step-label">{step.label}</span>
                    <span className="trace-step-time">{formatElapsed(elapsed)}</span>
                    {showTraceDetail && step.detail ? (
                      <pre className="trace-step-detail">{step.detail}</pre>
                    ) : null}
                  </li>
                );
              })}
            </ul>
          </div>
        ) : null}

        {error ? <div className="error">{error}</div> : null}

        <textarea

          id="chat"

          className="chat-input"

          placeholder="Try: Play John Mayer · What are my playlists? · Create a playlist called Focus"

          value={input}

          onChange={(e) => setInput(e.target.value)}

          disabled={sending}

          rows={3}

        />

        <div className="btn-row">

          <button type="button" onClick={() => void sendChat()} disabled={sending || !input.trim()}>

            {sending ? "Working…" : "Send"}

          </button>

        </div>

        {llm?.provider === "ollama" ? (

          <p className="hint">First reply after a backend restart can be slow while the model loads.</p>

        ) : null}

      </section>



      <details className="panel settings-disclosure">

        <summary className="settings-summary">

          <span className="settings-summary-title">Model &amp; Spotify</span>

          <span className="settings-summary-hint">LLM, account, playback device</span>

        </summary>



        <div className="settings-body">

          {llm ? (

            <div className="settings-block">

              <div className="settings-block-head">

                <span className="badge">{llm.provider === "gemini" ? "Gemini" : "Ollama"}</span>

                <span className={llm.reachable ? "badge ok" : "badge"}>{llm.reachable ? "OK" : "Issue"}</span>

              </div>

              <div className="control-row">

                <label htmlFor="llm-backend" className="control-label">

                  Backend

                </label>

                <select

                  id="llm-backend"

                  value={llmPick}

                  onChange={(e) => setLlmPick(e.target.value === "gemini" ? "gemini" : "ollama")}

                  disabled={llmSaving}

                >

                  <option value="ollama">Ollama (local)</option>

                  <option value="gemini">Gemini (API key)</option>

                </select>

                <button

                  type="button"

                  onClick={() => void applyLlmProvider()}

                  disabled={llmSaving || llmPick === (llm.provider === "gemini" ? "gemini" : "ollama")}

                >

                  Apply

                </button>

                <button

                  type="button"

                  className="secondary"

                  onClick={() => void resetLlmProvider()}

                  disabled={llmSaving || !llm.ui_override}

                >

                  Reset to .env

                </button>

                <button type="button" className="secondary" onClick={() => void refreshLlm()}>

                  Refresh status

                </button>

              </div>

              {llm.provider === "ollama" ? (

                <div className="control-row wrap">

                  <label htmlFor="ollama-model" className="control-label">

                    Model

                  </label>

                  <select

                    id="ollama-model"

                    value={ollamaModelSelect}

                    onChange={(e) => setOllamaModelSelect(e.target.value)}

                    disabled={llmSaving}

                  >

                    <option value="__env__">From .env ({llm.env_ollama_model?.trim() || "OLLAMA_MODEL"})</option>

                    {(llm.models ?? []).map((m) => (

                      <option key={m} value={m}>

                        {m}

                      </option>

                    ))}

                    <option value="__custom__">Custom…</option>

                  </select>

                  {ollamaModelSelect === "__custom__" ? (

                    <input

                      type="text"

                      value={ollamaCustomModel}

                      onChange={(e) => setOllamaCustomModel(e.target.value)}

                      placeholder="e.g. mistral:7b"

                      disabled={llmSaving}

                      className="control-input"

                      aria-label="Custom Ollama model tag"

                    />

                  ) : null}

                  <button type="button" onClick={() => void applyOllamaModel()} disabled={llmSaving || ollamaModelApplyDisabled}>

                    Apply model

                  </button>

                </div>

              ) : null}



              {llm.provider === "gemini" ? (

                <div className="control-row wrap">

                  <label htmlFor="gemini-model" className="control-label">

                    Model

                  </label>

                  <select

                    id="gemini-model"

                    value={geminiModelSelect}

                    onChange={(e) => setGeminiModelSelect(e.target.value)}

                    disabled={llmSaving || !llm.reachable}

                  >

                    <option value="__env__">From .env ({llm.env_gemini_model?.trim() || "GEMINI_MODEL"})</option>

                    {(llm.models ?? []).map((m) => (

                      <option key={m} value={m}>

                        {m}

                      </option>

                    ))}

                    <option value="__custom__">Custom…</option>

                  </select>

                  {geminiModelSelect === "__custom__" ? (

                    <input

                      type="text"

                      value={geminiCustomModel}

                      onChange={(e) => setGeminiCustomModel(e.target.value)}

                      placeholder="e.g. gemini-2.5-pro"

                      disabled={llmSaving}

                      className="control-input"

                      aria-label="Custom Gemini model name"

                    />

                  ) : null}

                  <button type="button" onClick={() => void applyGeminiModel()} disabled={llmSaving || geminiModelApplyDisabled}>

                    Apply model

                  </button>

                </div>

              ) : null}

              <p className="meta-line">

                {llm.provider === "gemini" ? (

                  <>

                    <code>{llm.configured_model || "—"}</code>

                    {llm.gemini_model_ui_override ? (

                      <>

                        {" "}

                        · override (env <code>{llm.env_gemini_model || "—"}</code>)

                      </>

                    ) : null}

                    {llm.ui_override ? (

                      <>

                        {" "}

                        · UI override (env: <code>{llm.env_provider ?? "ollama"}</code>)

                      </>

                    ) : null}

                  </>

                ) : (

                  <>

                    <code>{llm.configured_host || "—"}</code> · <code>{llm.configured_model || "—"}</code>

                    {llm.ollama_model_ui_override ? (

                      <>

                        {" "}

                        · override (env <code>{llm.env_ollama_model || "—"}</code>)

                      </>

                    ) : null}

                  </>

                )}

                {llm.reachable && llm.model_installed === false ? (

                  <span className="meta-warn">

                    {" "}

                    — {llm.provider === "gemini" ? "Check GEMINI_MODEL." : `Try: ollama pull ${llm.configured_model || "gemma2:2b"}`}

                  </span>

                ) : null}

              </p>

              {!llm.reachable ? (

                <p className="error tight">

                  {llm.error ?? "Unreachable"}

                  {llm.provider === "gemini" ? " Set GEMINI_API_KEY / LLM_PROVIDER in backend/.env." : " Start Ollama or set OLLAMA_HOST."}

                </p>

              ) : null}

            </div>

          ) : null}



          <div className="settings-block">

            <div className="settings-block-head">

              <span className="badge">Account</span>

              <span className={session?.signed_in ? "badge ok" : "badge"}>

                {session?.signed_in ? "Signed in" : "Not connected"}

              </span>

            </div>

            <div className="control-row">

              <a className="btn-link primary" href="/login">

                Connect Spotify

              </a>

              <button type="button" className="secondary" onClick={() => void logout()} disabled={!session?.signed_in}>

                Sign out

              </button>

            </div>
            {session?.signed_in && session.spotify_playlist_write_ok === false ? (
              <p className="scope-warn">
                This login is missing <code>playlist-modify-*</code> on the token. Click <strong>Connect Spotify</strong>{" "}
                again — the consent screen will open so you can approve all scopes.
              </p>
            ) : null}
            {session?.signed_in && session.spotify_playlist_write_ok === null ? (
              <p className="hint tight">
                Use <strong>Connect Spotify</strong> once more to refresh this install (older logins did not save granted scopes).
              </p>
            ) : null}
          </div>

          <div className="settings-block">

            <div className="settings-block-head">

              <span className="badge">Device</span>

            </div>

            <div className="control-row wrap">

              <label htmlFor="device" className="control-label">

                Playback

              </label>

              <select

                id="device"

                value={deviceId}

                onChange={(e) => setDeviceId(e.target.value)}

                disabled={!session?.signed_in || loadingDevices}

              >

                <option value="">{loadingDevices ? "Loading…" : "Select device"}</option>

                {deviceOptions.map((o) => (

                  <option key={o.value} value={o.value}>

                    {o.label}

                  </option>

                ))}

              </select>

              <button type="button" className="secondary" onClick={() => void refreshDevices()} disabled={!session?.signed_in}>

                Refresh

              </button>

              <button type="button" onClick={() => void saveDevice()} disabled={!session?.signed_in || !deviceId}>

                Save

              </button>

            </div>

            <p className="hint tight">Open Spotify on this machine so a Connect device appears. Playback uses the saved device.</p>

          </div>

        </div>

      </details>

    </div>

  );

}

