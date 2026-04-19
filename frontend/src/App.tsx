import { useCallback, useEffect, useMemo, useState } from "react";



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



type LlmStatus = {

  provider?: string;

  env_provider?: string;

  ui_override?: boolean;

  configured_host?: string;

  configured_model: string;

  env_ollama_model?: string;

  ollama_model_ui_override?: boolean;

  reachable: boolean;

  models: string[] | null;

  model_installed?: boolean;

  error: string | null;

};



const OLLAMA_MODEL_PRESETS = ["gemma2:2b", "gemma2:9b", "llama3.1:8b", "deepseek-coder-v2"] as const;



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

  const [trace, setTrace] = useState("");

  const [liveReply, setLiveReply] = useState("");

  const [llm, setLlm] = useState<LlmStatus | null>(null);

  const [llmPick, setLlmPick] = useState<"ollama" | "gemini">("ollama");

  const [llmSaving, setLlmSaving] = useState(false);

  const [ollamaModelSelect, setOllamaModelSelect] = useState("__env__");

  const [ollamaCustomModel, setOllamaCustomModel] = useState("");



  const refreshLlm = useCallback(async () => {

    try {

      const data = await readJson<LlmStatus>(await fetch("/api/llm"));

      setLlm(data);

      const p = data.provider === "gemini" ? "gemini" : "ollama";

      setLlmPick(p);

      const eff = (data.configured_model || "").trim();

      const fromEnv = !data.ollama_model_ui_override;

      if (fromEnv) {

        setOllamaModelSelect("__env__");

        setOllamaCustomModel(eff || (data.env_ollama_model ?? "").trim());

      } else if ((OLLAMA_MODEL_PRESETS as readonly string[]).includes(eff)) {

        setOllamaModelSelect(eff);

        setOllamaCustomModel(eff);

      } else {

        setOllamaModelSelect("__custom__");

        setOllamaCustomModel(eff);

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

    setTrace("");

    setLiveReply("");

    setInput("");

    setMessages((m) => [...m, { role: "user", text }]);

    const controller = new AbortController();

    const chatTimeoutMs = 900_000;

    const timeoutId = window.setTimeout(() => controller.abort(), chatTimeoutMs);



    const appendTrace = (line: string) => {

      setTrace((prev) => (prev ? `${prev}\n` : "") + line);

    };



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

          case "status":

            appendTrace(String(j.message ?? ""));

            break;

          case "round":

            appendTrace(`Round ${String(j.step ?? "?")} / ${String(j.max ?? "?")}`);

            break;

          case "llm_delta":

            reply += String(j.text ?? "");

            setLiveReply(reply);

            break;

          case "tool_start":

            appendTrace(`→ Tool: ${String(j.name ?? "")}…`);

            break;

          case "tool_done":

            appendTrace(`  ✓ ${String(j.name ?? "")} ${j.preview ? `(${String(j.preview)})` : ""}`);

            break;

          case "final":

            reply = String(j.text ?? "");

            setLiveReply(reply);

            break;

          case "error":

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

      if (streamOk) setTrace("");

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



  const statusChips = useMemo(() => {
    const bits: string[] = [];
    if (session?.signed_in) bits.push("Spotify");
    else bits.push("Spotify off");
    if (session?.signed_in) {
      if (session.spotify_playlist_write_ok === true) bits.push("playlist edit OK");
      else if (session.spotify_playlist_write_ok === false) bits.push("reconnect for playlists");
    }
    if (llm?.provider === "gemini") bits.push("Gemini");
    else if (llm?.provider === "ollama") bits.push("Ollama");
    if (session?.device_id) bits.push("device saved");
    return bits.join(" · ");
  }, [session?.signed_in, session?.device_id, session?.spotify_playlist_write_ok, llm?.provider]);



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

        <div className="messages" role="log" aria-relevant="additions">

          {messages.map((m, i) => (

            <div key={i} className={`bubble ${m.role}`}>

              {m.text}

            </div>

          ))}

        </div>

        {sending && (liveReply || trace) ? (

          <div className="bubble assistant streaming" aria-live="polite">

            {liveReply || "…"}

          </div>

        ) : null}

        {trace ? (

          <pre className="trace-log" aria-live="polite">

            {trace}

          </pre>

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

                    {OLLAMA_MODEL_PRESETS.map((m) => (

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

              <p className="meta-line">

                {llm.provider === "gemini" ? (

                  <>

                    <code>{llm.configured_model || "—"}</code>

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

