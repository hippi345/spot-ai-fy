import { useCallback, useEffect, useState } from "react";

import {
  fetchSetupStatus,
  probeOllama,
  saveLlmSetup,
  saveSpotifyApp,
  type SetupStatus,
} from "./lib/api";

type Props = {
  onComplete: () => void;
  onDismiss?: () => void;
  onSettingsSaved?: () => void;
  allowDismiss: boolean;
  closeOnComplete?: boolean;
};

export function SetupWizard({
  onComplete,
  onDismiss,
  onSettingsSaved,
  allowDismiss,
  closeOnComplete = false,
}: Props) {
  const [step, setStep] = useState<1 | 2>(1);
  const [status, setStatus] = useState<SetupStatus | null>(null);
  const [clientId, setClientId] = useState("");
  const [provider, setProvider] = useState<"ollama" | "gemini">("ollama");
  const [geminiKey, setGeminiKey] = useState("");
  const [ollamaHost, setOllamaHost] = useState("http://127.0.0.1:11434");
  const [ollamaModel, setOllamaModel] = useState("");
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [geminiModel, setGeminiModel] = useState("gemini-2.5-flash");
  const [geminiModels, setGeminiModels] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [ollamaAllowPublic, setOllamaAllowPublic] = useState(false);
  const [smallModelMode, setSmallModelMode] = useState<"auto" | "on" | "off">("auto");
  const [savedNotice, setSavedNotice] = useState<string | null>(null);
  const [cpuHint, setCpuHint] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const s = await fetchSetupStatus();
    setStatus(s);
    setProvider(s.provider === "gemini" ? "gemini" : "ollama");
    if (s.ollama_host) setOllamaHost(s.ollama_host);
    if (s.ollama_model) setOllamaModel(s.ollama_model);
    else if (!ollamaModel) setOllamaModel("qwen3:4b-instruct");
    if (s.gemini_model) setGeminiModel(s.gemini_model);
    if (closeOnComplete && s.setup_complete) onComplete();
    if (s.setup_complete) setStep(1);
    else if (s.spotify_configured && s.spotify_signed_in) setStep(2);
  }, [onComplete, closeOnComplete]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const copyRedirect = async () => {
    if (!status?.redirect_uri) return;
    try {
      await navigator.clipboard.writeText(status.redirect_uri);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setError("Could not copy — select the URI and copy manually.");
    }
  };

  const saveSpotify = async () => {
    setBusy(true);
    setError(null);
    try {
      const out = await saveSpotifyApp(clientId.trim());
      await refresh();
      setStep(2);
      if (!out.redirect_uri) return;
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not save Spotify app");
    } finally {
      setBusy(false);
    }
  };

  const connectSpotify = () => {
    window.location.href = "/login";
  };

  const detectOllama = async () => {
    setBusy(true);
    setError(null);
    try {
      const out = await probeOllama(ollamaHost.trim(), ollamaAllowPublic);
      setOllamaModels(out.models);
      if (!out.reachable) setError(out.error ?? "Ollama is not reachable");
      else if (out.models.length && !ollamaModel) setOllamaModel(out.models[0]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Ollama probe failed");
    } finally {
      setBusy(false);
    }
  };

  const saveLlm = async () => {
    setBusy(true);
    setError(null);
    try {
      const out = await saveLlmSetup({
        provider,
        gemini_api_key: provider === "gemini" ? geminiKey.trim() : undefined,
        ollama_host: provider === "ollama" ? ollamaHost.trim() : undefined,
        ollama_model: provider === "ollama" ? ollamaModel.trim() : undefined,
        gemini_model: provider === "gemini" ? geminiModel.trim() : undefined,
        ollama_allow_public: provider === "ollama" ? ollamaAllowPublic : undefined,
        ollama_small_model_mode: provider === "ollama" ? smallModelMode : undefined,
        test: true,
      });
      if (out.models?.length) {
        if (provider === "ollama") setOllamaModels(out.models);
        else setGeminiModels(out.models);
      }
      const profile = (out as { ollama_cpu_profile?: { message?: string | null } }).ollama_cpu_profile;
      setCpuHint(profile?.message ?? null);
      setSavedNotice("Saved — settings tested successfully.");
      await refresh();
      onSettingsSaved?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : "LLM setup failed");
    } finally {
      setBusy(false);
    }
  };

  const redirectUri = status?.redirect_uri ?? "http://127.0.0.1:8765/callback";

  return (
    <div className="setup-overlay" role="dialog" aria-modal="true" aria-labelledby="setup-title">
      <div className="setup-card panel">
        <div className="setup-head">
          <h2 id="setup-title">{status?.setup_complete ? "Settings" : "First-time setup"}</h2>
          {allowDismiss && onDismiss ? (
            <button type="button" className="setup-dismiss" onClick={onDismiss}>
              Close
            </button>
          ) : null}
        </div>
        <p className="hint">
          Connect Spotify and configure a model — no <code>.env</code> editing required. Environment variables still
          override saved settings.
        </p>

        {step === 1 ? (
          <div className="setup-step">
            <h3>Step 1 — Spotify app</h3>
            <p className="hint">
              In the{" "}
              <a href="https://developer.spotify.com/dashboard" target="_blank" rel="noreferrer">
                Spotify Developer Dashboard
              </a>
              , open your app → Settings → Redirect URIs and add this exact URI:
            </p>
            <div className="setup-uri-row">
              <code className="setup-uri">{redirectUri}</code>
              <button type="button" onClick={() => void copyRedirect()} disabled={busy}>
                {copied ? "Copied" : "Copy"}
              </button>
            </div>
            <label className="control-label" htmlFor="spotify-client-id">Client ID</label>
            <input
              id="spotify-client-id"
              className="setup-input"
              value={clientId}
              onChange={(e) => setClientId(e.target.value)}
              placeholder="Paste your Spotify Client ID"
              autoComplete="off"
            />
            <div className="btn-row">
              <button type="button" onClick={() => void saveSpotify()} disabled={busy || !clientId.trim()}>
                Save Client ID
              </button>
              <button
                type="button"
                onClick={connectSpotify}
                disabled={busy || !(status?.spotify_configured || clientId.trim())}
              >
                Connect Spotify
              </button>
            </div>
            {status?.spotify_signed_in ? (
              <p className="hint ok-text">Spotify signed in — continue to step 2.</p>
            ) : null}
            <button type="button" className="setup-link" onClick={() => setStep(2)} disabled={!status?.spotify_signed_in}>
              Continue to LLM setup →
            </button>
          </div>
        ) : (
          <div className="setup-step">
            <h3>Step 2 — Language model</h3>
            <div className="control-row">
              <label className="control-label" htmlFor="llm-provider">Provider</label>
              <select
                id="llm-provider"
                value={provider}
                onChange={(e) => setProvider(e.target.value === "gemini" ? "gemini" : "ollama")}
                disabled={busy}
              >
                <option value="ollama">Ollama (local)</option>
                <option value="gemini">Gemini (API key)</option>
              </select>
            </div>
            {provider === "ollama" ? (
              <>
                <label className="control-label" htmlFor="ollama-host">Ollama URL</label>
                <input
                  id="ollama-host"
                  className="setup-input"
                  value={ollamaHost}
                  onChange={(e) => setOllamaHost(e.target.value)}
                />
                <div className="btn-row">
                  <button type="button" onClick={() => void detectOllama()} disabled={busy}>
                    Detect models
                  </button>
                </div>
                <label className="control-label">
                  <input
                    type="checkbox"
                    checked={ollamaAllowPublic}
                    onChange={(e) => setOllamaAllowPublic(e.target.checked)}
                  />
                  Allow Ollama on the public internet (not recommended)
                </label>
                <label className="control-label" htmlFor="ollama-model">Model</label>
                {ollamaModels.length ? (
                  <select
                    id="ollama-model"
                    value={ollamaModel}
                    onChange={(e) => setOllamaModel(e.target.value)}
                    disabled={busy}
                  >
                    {ollamaModels.map((m) => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    id="ollama-model"
                    className="setup-input"
                    value={ollamaModel}
                    onChange={(e) => setOllamaModel(e.target.value)}
                    placeholder="e.g. qwen3:4b-instruct (lighter: qwen2.5:3b-instruct)"
                  />
                )}
                <label className="control-label" htmlFor="small-model-mode">Small-model tool set</label>
                <select
                  id="small-model-mode"
                  value={smallModelMode}
                  onChange={(e) => setSmallModelMode(e.target.value as "auto" | "on" | "off")}
                  disabled={busy}
                >
                  <option value="auto">Auto (models under ~8B)</option>
                  <option value="on">Always on</option>
                  <option value="off">Off (full tool list)</option>
                </select>
              </>
            ) : (
              <>
                <label className="control-label" htmlFor="gemini-key">Gemini API key</label>
                <input
                  id="gemini-key"
                  className="setup-input"
                  type="password"
                  value={geminiKey}
                  onChange={(e) => setGeminiKey(e.target.value)}
                  placeholder={status?.gemini_api_key_masked ? "Key saved (enter to replace)" : "AIza…"}
                  autoComplete="off"
                />
                <label className="control-label" htmlFor="gemini-model">Model</label>
                {geminiModels.length ? (
                  <select
                    id="gemini-model"
                    value={geminiModel}
                    onChange={(e) => setGeminiModel(e.target.value)}
                    disabled={busy}
                  >
                    {geminiModels.map((m) => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    id="gemini-model"
                    className="setup-input"
                    value={geminiModel}
                    onChange={(e) => setGeminiModel(e.target.value)}
                  />
                )}
              </>
            )}
            <div className="btn-row">
              <button type="button" onClick={() => void saveLlm()} disabled={busy}>
                Save &amp; test
              </button>
              <button type="button" className="setup-link" onClick={() => setStep(1)}>
                ← Back
              </button>
            </div>
            {savedNotice ? <p className="hint ok-text">{savedNotice}</p> : null}
            {cpuHint ? <p className="hint">{cpuHint}</p> : null}
          </div>
        )}

        {error ? <div className="error">{error}</div> : null}
        {step === 2 && provider === "ollama" && ollamaHost.trim() && !ollamaModel.trim() ? (
          <p className="hint">Select or enter an Ollama model name (try Detect models).</p>
        ) : null}
        {status?.llm_error && step === 2 && !(provider === "ollama" && ollamaHost.trim()) ? (
          <p className="hint">{status.llm_error}</p>
        ) : null}
      </div>
    </div>
  );
}
