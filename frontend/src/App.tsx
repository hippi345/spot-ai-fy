import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";

import { DesktopTitleBar } from "./components/DesktopTitleBar";
import { LiquidBackground } from "./components/LiquidBackground";
import { ModelSpotifySettings } from "./components/ModelSpotifySettings";
import { NowPlayingBar, type NowPlayingBarHandle } from "./components/NowPlayingBar";
import { SettingsSheet } from "./components/SettingsSheet";
import { IconClose, IconGear } from "./components/icons/AppIcons";
import { nowPlayingUsesMock } from "./lib/nowPlaying";
import { SetupWizard } from "./SetupWizard";
import { fetchSetupStatus, type SetupStatus } from "./lib/api";
import {
  buildChatStreamRequestBody,
  loadChatSession,
  saveChatSession,
  startNewChatSession,
} from "./lib/chatSession";
import { FRIENDLY_SPOTIFY_GUIDANCE, isUnpersistedAssistantFallback } from "./lib/chatMessages";
import { isChatBlockedBySetup } from "./lib/setupGate";
import {
  type TraceStep,
  reduceTraceFinishAllRunning,
  reduceTraceFinishStep,
  reduceTracePushStep,
} from "./lib/chatTrace";
import { shouldSendChatOnEnter } from "./lib/chatInputKeyboard";
import {
  type LlmProviderId,
  type LlmStatus,
  normalizeLlmProvider,
  providerLabel,
  providerUsesApiKey,
} from "./lib/llmTypes";



type Session = {
  signed_in: boolean;
  device_id: string | null;
  spotify_granted_scopes?: string | null;
  spotify_playlist_write_ok?: boolean | null;
  spotify_missing_scopes?: string[] | null;
  spotify_reauth_recommended?: boolean;
};



type SpotifyDevice = {

  id: string;

  name: string;

  is_active: boolean;

  type: string;

};



type ChatMessage = { role: "user" | "assistant"; text: string; trace?: TraceStep[] };






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



  const nowPlayingRef = useRef<NowPlayingBarHandle | null>(null);

  const [input, setInput] = useState("");

  const [sending, setSending] = useState(false);

  const [conversationId, setConversationId] = useState<string>(() => loadChatSession().conversationId);

  const [messages, setMessages] = useState<ChatMessage[]>(() =>
    loadChatSession().messages.map((m) => ({ role: m.role, text: m.text })),
  );

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

  useEffect(() => {
    saveChatSession({
      conversationId,
      messages: messages.map((m) => ({ role: m.role, text: m.text })),
    });
  }, [conversationId, messages]);

  const beginNewChat = useCallback(() => {
    const fresh = startNewChatSession();
    setConversationId(fresh.conversationId);
    setMessages([]);
    setLiveReply("");
    setTraceSteps([]);
    setError(null);
    setInput("");
  }, []);

  const [llm, setLlm] = useState<LlmStatus | null>(null);

  const [llmPick, setLlmPick] = useState<LlmProviderId>("ollama");

  const [llmSaving, setLlmSaving] = useState(false);

  const [modelSelect, setModelSelect] = useState("__env__");

  const [modelCustom, setModelCustom] = useState("");

  const [apiKeyDraft, setApiKeyDraft] = useState("");

  const [apiKeySaving, setApiKeySaving] = useState(false);

  const chatComposingRef = useRef(false);

  const [setupComplete, setSetupComplete] = useState(false);
  const [showSetupWizard, setShowSetupWizard] = useState(false);
  const [wizardAutoOpened, setWizardAutoOpened] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [bgArtUrl, setBgArtUrl] = useState<string | null>(null);
  const [bgHasTrack, setBgHasTrack] = useState(false);
  const [bgPlaying, setBgPlaying] = useState(false);
  const [setupStatus, setSetupStatus] = useState<SetupStatus | null>(null);
  const [streamStalled, setStreamStalled] = useState(false);
  const streamIdleMs = 45_000;

  const refreshSetup = useCallback(async () => {
    try {
      const s = await fetchSetupStatus();
      setSetupStatus(s);
      const ready = !isChatBlockedBySetup(s) && s.spotify_configured;
      setSetupComplete(ready);
      if (!ready) {
        setShowSetupWizard(true);
        setWizardAutoOpened(true);
      }
    } catch {
      setSetupComplete(false);
      setShowSetupWizard(true);
    }
  }, []);

  const syncModelPickers = useCallback((data: LlmStatus) => {
    const p = normalizeLlmProvider(data.provider);
    setLlmPick(p);
    const eff = (data.configured_model || "").trim();
    const available = data.models ?? [];
    const fromEnv =
      p === "ollama"
        ? !data.ollama_model_ui_override
        : p === "gemini"
          ? !data.gemini_model_ui_override
          : p === "openai"
            ? !data.openai_model_ui_override
            : p === "anthropic"
              ? !data.anthropic_model_ui_override
              : !data.xai_model_ui_override;
    if (fromEnv) {
      setModelSelect("__env__");
      const envFallback =
        p === "gemini"
          ? data.env_gemini_model
          : p === "openai"
            ? data.env_openai_model
            : p === "anthropic"
              ? data.env_anthropic_model
              : p === "xai"
                ? data.env_xai_model
                : data.env_ollama_model;
      setModelCustom(eff || (envFallback ?? "").trim());
    } else if (available.includes(eff)) {
      setModelSelect(eff);
      setModelCustom(eff);
    } else {
      setModelSelect("__custom__");
      setModelCustom(eff);
    }
    setApiKeyDraft("");
  }, []);

  const refreshLlm = useCallback(async () => {
    try {
      const data = await readJson<LlmStatus>(await fetch("/api/llm"));
      setLlm(data);
      syncModelPickers(data);
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
  }, [syncModelPickers]);



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



  const applyProviderModel = async () => {
    if (!llm) return;
    const provider = normalizeLlmProvider(llm.provider);
    setLlmSaving(true);
    setError(null);
    try {
      if (modelSelect === "__env__") {
        if (provider === "ollama") {
          await readJson(await fetch("/api/llm/ollama-model", { method: "DELETE" }));
        } else if (provider === "gemini") {
          await readJson(await fetch("/api/llm/gemini-model", { method: "DELETE" }));
        } else {
          await readJson(await fetch(`/api/llm/model?provider=${provider}`, { method: "DELETE" }));
        }
        setBanner("Model from .env again.");
      } else {
        const tag = modelSelect === "__custom__" ? modelCustom.trim() : modelSelect.trim();
        if (!tag) throw new Error("Enter a model name.");
        if (provider === "ollama") {
          await readJson(
            await fetch("/api/llm/ollama-model", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ model: tag }),
            }),
          );
        } else if (provider === "gemini") {
          await readJson(
            await fetch("/api/llm/gemini-model", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ model: tag }),
            }),
          );
        } else {
          await readJson(
            await fetch("/api/llm/model", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ provider, model: tag }),
            }),
          );
        }
        setBanner(`${providerLabel(provider)}: ${tag}.`);
      }
      await refreshLlm();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not save model");
    } finally {
      setLlmSaving(false);
    }
  };

  const modelApplyDisabled = useMemo(() => {
    if (!llm) return true;
    const provider = normalizeLlmProvider(llm.provider);
    const overrideActive =
      provider === "ollama"
        ? Boolean(llm.ollama_model_ui_override)
        : provider === "gemini"
          ? Boolean(llm.gemini_model_ui_override)
          : provider === "openai"
            ? Boolean(llm.openai_model_ui_override)
            : provider === "anthropic"
              ? Boolean(llm.anthropic_model_ui_override)
              : Boolean(llm.xai_model_ui_override);
    if (modelSelect === "__env__") return !overrideActive;
    if (modelSelect === "__custom__") {
      const t = modelCustom.trim();
      if (!t) return true;
      return Boolean(overrideActive && t === (llm.configured_model || "").trim());
    }
    return Boolean(overrideActive && modelSelect === (llm.configured_model || "").trim());
  }, [llm, modelSelect, modelCustom]);

  const saveApiKey = async () => {
    if (!llm) return;
    const provider = normalizeLlmProvider(llm.provider);
    if (!providerUsesApiKey(provider)) return;
    const key = apiKeyDraft.trim();
    if (!key) return;
    setApiKeySaving(true);
    setError(null);
    try {
      await readJson(
        await fetch("/api/llm/api-key", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ provider, api_key: key }),
        }),
      );
      setApiKeyDraft("");
      setBanner(`${providerLabel(provider)} API key saved locally.`);
      await refreshLlm();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not save API key");
    } finally {
      setApiKeySaving(false);
    }
  };



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

    void refreshSetup();

  }, [refreshSession, refreshLlm, refreshSetup]);



  useEffect(() => {

    if (session?.signed_in) void refreshDevices();

  }, [session?.signed_in, refreshDevices]);

  useEffect(() => {
    if (!session?.signed_in) {
      setBgArtUrl(null);
      setBgHasTrack(false);
      setBgPlaying(false);
    }
  }, [session?.signed_in]);

  const backgroundIdle =
    !session?.signed_in || !bgHasTrack;

  const handleBackgroundArtChange = useCallback(
    (info: { artUrl: string | null; hasTrack: boolean; isPlaying: boolean }) => {
      setBgArtUrl(info.artUrl);
      setBgHasTrack(info.hasTrack);
      setBgPlaying(info.isPlaying);
    },
    [],
  );



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
    const autoLabel = loadingDevices ? "Loading…" : "Auto (active device)";
    const auto = { value: "", label: autoLabel };
    return [
      auto,
      ...devices.map((d) => ({
        value: d.id,
        label: `${d.name} (${d.type})${d.is_active ? " · active" : ""}`,
      })),
    ];
  }, [devices, loadingDevices]);



  const saveDevice = async () => {
    setError(null);
    if (!deviceId) {
      await fetch("/api/device", { method: "DELETE" });
      await refreshSession();
      setBanner("Using auto (active Spotify device).");
      return;
    }

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

    const historyPayload = messages.map((m) => ({ role: m.role, content: m.text }));

    setSending(true);

    setError(null);

    setStreamStalled(false);

    setTraceSteps([]);

    setLiveReply("");

    setInput("");

    setMessages((m) => [...m, { role: "user", text }]);

    let traceAccum: TraceStep[] = [];

    const controller = new AbortController();

    const chatTimeoutMs = 900_000;

    const timeoutId = window.setTimeout(() => controller.abort(), chatTimeoutMs);

    let nextStepId = 1;
    const newStepId = () => nextStepId++;

    const pushStep = (step: Omit<TraceStep, "id" | "startedAt"> & { startedAt?: number }) => {
      const id = newStepId();
      const now = Date.now();
      const next = reduceTracePushStep(traceAccum, step, id, now);
      traceAccum = next;
      setTraceSteps(next);
      return id;
    };

    const finishStep = (id: number, patch?: Partial<TraceStep>) => {
      const now = Date.now();
      const next = reduceTraceFinishStep(traceAccum, id, patch, now);
      traceAccum = next;
      setTraceSteps(next);
    };

    const finishAllRunning = () => {
      const now = Date.now();
      const next = reduceTraceFinishAllRunning(traceAccum, now);
      traceAccum = next;
      setTraceSteps(next);
    };

    const toolStepIdByName = new Map<string, number>();



    let lastStreamEventAt = Date.now();
    let idleTimer: number | undefined;

    const bumpStreamActivity = () => {
      lastStreamEventAt = Date.now();
      setStreamStalled(false);
    };

    const armIdleTimer = () => {
      if (idleTimer !== undefined) window.clearInterval(idleTimer);
      idleTimer = window.setInterval(() => {
        if (Date.now() - lastStreamEventAt >= streamIdleMs) {
          setStreamStalled(true);
        }
      }, 2000);
    };

    armIdleTimer();

    try {

      const res = await fetch("/api/chat/stream", {

        method: "POST",

        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },

        body: JSON.stringify(buildChatStreamRequestBody(text, historyPayload, conversationId)),

        signal: controller.signal,

      });



      if (!res.ok) {

        const errText = await res.text();

        let msg = errText || res.statusText;

        try {
          const j = JSON.parse(errText) as { detail?: unknown };
          if (typeof j.detail === "string") {
            msg = j.detail;
          } else if (Array.isArray(j.detail)) {
            msg = j.detail
              .map((item) => {
                if (!item || typeof item !== "object") return "";
                const loc = Array.isArray((item as { loc?: unknown }).loc)
                  ? (item as { loc: unknown[] }).loc.join(".")
                  : "";
                const part = String((item as { msg?: unknown }).msg ?? "");
                return loc ? `${loc}: ${part}` : part;
              })
              .filter(Boolean)
              .join("; ");
          }
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
        bumpStreamActivity();

        const typ = String(j.type || "");

        switch (typ) {

          case "keepalive": {
            const msg = String(j.message ?? "Still working…");
            const sid = pushStep({ kind: "status", label: msg, status: "done" });
            finishStep(sid);
            break;
          }

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



      const emptyish =
        !reply.trim() ||
        reply.trim() === "No response from model." ||
        isUnpersistedAssistantFallback(reply);

      const assistantText = emptyish ? FRIENDLY_SPOTIFY_GUIDANCE : reply;
      const traceCopy = traceAccum.map((s) => ({ ...s }));
      setMessages((m) => [...m, { role: "assistant", text: assistantText, trace: traceCopy }]);
      setTraceSteps([]);

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

      if (idleTimer !== undefined) window.clearInterval(idleTimer);

      setSending(false);

      setLiveReply("");

      setStreamStalled(false);

      void nowPlayingRef.current?.refresh();

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

    beginNewChat();

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

  const traceIcon = (step: TraceStep): "running" | "tool" | "round" | "dot" => {
    if (step.status === "running") return "running";
    if (step.kind === "tool") return "tool";
    if (step.kind === "round") return "round";
    return "dot";
  };

  const showTracePanel = sending && traceSteps.length > 0;

  const statusChips = useMemo(() => {
    let spotifyLabel = "Spotify — Not connected";
    if (session?.signed_in) spotifyLabel = "Spotify — Connected";
    else if (setupStatus?.spotify_configured) spotifyLabel = "Spotify — Client ID saved, sign in to connect";
    const llmName = providerLabel(
      normalizeLlmProvider(showSetupWizard ? setupStatus?.provider : llm?.provider),
    );
    const llmConnected = showSetupWizard
      ? Boolean(setupStatus?.llm_ready)
      : Boolean(setupStatus?.llm_ready ?? llm?.reachable);
    const llmLabel = `${llmName} — ${llmConnected ? "Connected" : "Not connected"}`;
    return `${spotifyLabel} · ${llmLabel}`;
  }, [session?.signed_in, llm?.provider, llm?.reachable, setupStatus, showSetupWizard]);



  const settingsPanelProps = {
    llm,
    llmPick,
    llmSaving,
    modelSelect,
    modelCustom,
    modelApplyDisabled,
    apiKeyDraft,
    apiKeySaving,
    session,
    deviceId,
    deviceOptions,
    loadingDevices,
    onLlmPickChange: setLlmPick,
    onApplyLlmProvider: () => void applyLlmProvider(),
    onResetLlmProvider: () => void resetLlmProvider(),
    onRefreshLlm: () => void refreshLlm(),
    onModelSelectChange: setModelSelect,
    onModelCustomChange: setModelCustom,
    onApplyModel: () => void applyProviderModel(),
    onApiKeyDraftChange: setApiKeyDraft,
    onSaveApiKey: () => void saveApiKey(),
    onOpenSetupWizard: () => {
      setSettingsOpen(false);
      setShowSetupWizard(true);
    },
    onLogout: () => void logout(),
    onDeviceIdChange: setDeviceId,
    onRefreshDevices: () => void refreshDevices(),
    onSaveDevice: () => void saveDevice(),
  };

  const handleChatKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (shouldSendChatOnEnter(e) && !chatComposingRef.current) {
      e.preventDefault();
      void sendChat();
    }
  };

  return (

    <>
      <LiquidBackground idle={backgroundIdle} artUrl={bgArtUrl} paused={!bgPlaying} />

      <div className="app-shell">

      <DesktopTitleBar />

      <div className="app">

      <header className="app-header glass-panel">

        <div className="app-header-main">

          <h1>Spot-AI-fy</h1>

          <p className="tagline">Ask Spotify in plain language — search, playlists, playback.</p>

        </div>

        <div className="app-header-actions">
          <p className="status-chips" aria-live="polite">
            {statusChips}
          </p>
          <button
            type="button"
            className="header-gear-btn secondary"
            aria-label="Model and Spotify settings"
            onClick={() => setSettingsOpen(true)}
          >
            <IconGear className="header-gear-icon" />
          </button>
        </div>

      </header>



      {session?.spotify_reauth_recommended ? (
        <div className="banner banner-warn" role="status">
          Spotify is missing permissions ({session.spotify_missing_scopes?.slice(0, 4).join(", ")}
          {session.spotify_missing_scopes && session.spotify_missing_scopes.length > 4 ? "…" : ""}).{" "}
          <a href="/login">Re-authorize Spotify</a>
        </div>
      ) : null}



      {banner ? (

        <div className="banner" role="status">

          <span>{banner}</span>

          <button type="button" className="banner-dismiss" onClick={() => setBanner(null)} aria-label="Dismiss">

            <IconClose />

          </button>

        </div>

      ) : null}

      {showSetupWizard ? (
        <SetupWizard
          allowDismiss={setupComplete}
          closeOnComplete={wizardAutoOpened}
          onDismiss={() => {
            setShowSetupWizard(false);
            setWizardAutoOpened(false);
          }}
          onSettingsSaved={(patch) => {
            const nextProvider = patch?.provider;
            if (nextProvider === "gemini" || nextProvider === "ollama") {
              setSetupStatus((prev) => (prev ? { ...prev, provider: nextProvider } : prev));
            }
            void refreshLlm();
            void refreshSetup();
          }}
          onComplete={() => {
            setSetupComplete(true);
            setShowSetupWizard(false);
            setWizardAutoOpened(false);
            void refreshLlm();
            void refreshSession();
            void refreshSetup();
          }}
        />
      ) : null}

      <section
        className={`panel glass-panel chat-panel${setupComplete ? "" : " chat-panel--blocked"}`}
        aria-labelledby="chat-heading"
      >

        <div className="chat-panel-head">

          <h2 id="chat-heading" className="panel-title">

            Chat

          </h2>

          <button
            type="button"
            className="secondary"
            onClick={beginNewChat}
            disabled={sending}
          >
            New chat
          </button>

        </div>

        <div className="chat-panel-mid">

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

              {m.role === "assistant" &&
              m.trace &&
              m.trace.filter((s) => s.kind === "tool").length > 0 ? (
                <details className="message-trace">
                  <summary>
                    Actions taken ({m.trace.filter((s) => s.kind === "tool").length})
                  </summary>
                  <ul className="trace-steps compact">
                    {m.trace.filter((s) => s.kind === "tool").map((step) => (
                      <li key={step.id}>
                        {step.label}
                        {step.detail ? ` — ${step.detail.slice(0, 120)}` : ""}
                      </li>
                    ))}
                  </ul>
                </details>
              ) : null}

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
                    <span
                      className={`trace-step-icon trace-step-icon--${traceIcon(step)}`}
                      aria-hidden="true"
                    />
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

        {streamStalled && sending ? (
          <div className="error">
            No response for {Math.round(streamIdleMs / 1000)}s — the model may still be thinking.{" "}
            <button type="button" onClick={() => void sendChat()}>
              Retry
            </button>
          </div>
        ) : null}

        </div>

        <div className="chat-panel-foot">

        <NowPlayingBar
          ref={nowPlayingRef}
          signedIn={nowPlayingUsesMock() || Boolean(session?.signed_in)}
          onBackgroundArtChange={handleBackgroundArtChange}
        />

        <textarea
          id="chat"
          className="chat-input"
          placeholder="Try: Play John Mayer · What are my playlists? · Create a playlist called Focus"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onCompositionStart={() => {
            chatComposingRef.current = true;
          }}
          onCompositionEnd={() => {
            chatComposingRef.current = false;
          }}
          onKeyDown={handleChatKeyDown}
          disabled={sending || !setupComplete}
          rows={3}
        />

        <div className="btn-row">

          <button
            type="button"
            onClick={() => void sendChat()}
            disabled={sending || !input.trim() || !setupComplete}
          >

            {sending ? "Working…" : "Send"}

          </button>

        </div>

        {llm?.provider === "ollama" ? (

          <p className="hint">First reply after a backend restart can be slow while the model loads.</p>

        ) : null}

        </div>

      </section>

      <SettingsSheet
        open={settingsOpen}
        title="Model & Spotify"
        onClose={() => setSettingsOpen(false)}
      >
        <ModelSpotifySettings {...settingsPanelProps} />
      </SettingsSheet>

      </div>
      </div>
    </>

  );

}

