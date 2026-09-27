import type { ReactNode } from "react";

type Session = {
  signed_in: boolean;
  spotify_playlist_write_ok?: boolean | null;
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

type DeviceOption = { value: string; label: string };

export type ModelSpotifySettingsProps = {
  llm: LlmStatus | null;
  llmPick: "ollama" | "gemini";
  llmSaving: boolean;
  ollamaModelSelect: string;
  ollamaCustomModel: string;
  geminiModelSelect: string;
  geminiCustomModel: string;
  ollamaModelApplyDisabled: boolean;
  geminiModelApplyDisabled: boolean;
  session: Session | null;
  deviceId: string;
  deviceOptions: DeviceOption[];
  loadingDevices: boolean;
  onLlmPickChange: (v: "ollama" | "gemini") => void;
  onApplyLlmProvider: () => void;
  onResetLlmProvider: () => void;
  onRefreshLlm: () => void;
  onOllamaModelSelectChange: (v: string) => void;
  onOllamaCustomModelChange: (v: string) => void;
  onApplyOllamaModel: () => void;
  onGeminiModelSelectChange: (v: string) => void;
  onGeminiCustomModelChange: (v: string) => void;
  onApplyGeminiModel: () => void;
  onOpenSetupWizard: () => void;
  onLogout: () => void;
  onDeviceIdChange: (v: string) => void;
  onRefreshDevices: () => void;
  onSaveDevice: () => void;
};

export function ModelSpotifySettings(props: ModelSpotifySettingsProps): ReactNode {
  const {
    llm,
    llmPick,
    llmSaving,
    ollamaModelSelect,
    ollamaCustomModel,
    geminiModelSelect,
    geminiCustomModel,
    ollamaModelApplyDisabled,
    geminiModelApplyDisabled,
    session,
    deviceId,
    deviceOptions,
    loadingDevices,
    onLlmPickChange,
    onApplyLlmProvider,
    onResetLlmProvider,
    onRefreshLlm,
    onOllamaModelSelectChange,
    onOllamaCustomModelChange,
    onApplyOllamaModel,
    onGeminiModelSelectChange,
    onGeminiCustomModelChange,
    onApplyGeminiModel,
    onOpenSetupWizard,
    onLogout,
    onDeviceIdChange,
    onRefreshDevices,
    onSaveDevice,
  } = props;

  return (
    <div className="settings-body settings-body--sheet">
      {llm ? (
        <div className="settings-block">
          <div className="settings-block-head">
            <span className="badge">{llm.provider === "gemini" ? "Gemini" : "Ollama"}</span>
            <span className={llm.reachable ? "badge ok" : "badge"}>{llm.reachable ? "OK" : "Issue"}</span>
          </div>
          <div className="control-row">
            <label htmlFor="llm-backend" className="control-label">Backend</label>
            <select
              id="llm-backend"
              value={llmPick}
              onChange={(e) => onLlmPickChange(e.target.value === "gemini" ? "gemini" : "ollama")}
              disabled={llmSaving}
            >
              <option value="ollama">Ollama (local)</option>
              <option value="gemini">Gemini (API key)</option>
            </select>
            <button
              type="button"
              onClick={() => void onApplyLlmProvider()}
              disabled={llmSaving || llmPick === (llm.provider === "gemini" ? "gemini" : "ollama")}
            >
              Apply
            </button>
            <button
              type="button"
              className="secondary"
              onClick={() => void onResetLlmProvider()}
              disabled={llmSaving || !llm.ui_override}
            >
              Reset to .env
            </button>
            <button type="button" className="secondary" onClick={() => void onRefreshLlm()}>
              Refresh status
            </button>
          </div>
          {llm.provider === "ollama" ? (
            <div className="control-row wrap">
              <label htmlFor="ollama-model" className="control-label">Model</label>
              <select
                id="ollama-model"
                value={ollamaModelSelect}
                onChange={(e) => onOllamaModelSelectChange(e.target.value)}
                disabled={llmSaving}
              >
                <option value="__env__">From .env ({llm.env_ollama_model?.trim() || "OLLAMA_MODEL"})</option>
                {(llm.models ?? []).map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
                <option value="__custom__">Custom…</option>
              </select>
              {ollamaModelSelect === "__custom__" ? (
                <input
                  type="text"
                  value={ollamaCustomModel}
                  onChange={(e) => onOllamaCustomModelChange(e.target.value)}
                  placeholder="e.g. mistral:7b"
                  disabled={llmSaving}
                  className="control-input"
                  aria-label="Custom Ollama model tag"
                />
              ) : null}
              <button
                type="button"
                onClick={() => void onApplyOllamaModel()}
                disabled={llmSaving || ollamaModelApplyDisabled}
              >
                Apply model
              </button>
            </div>
          ) : null}
          {llm.provider === "gemini" ? (
            <div className="control-row wrap">
              <label htmlFor="gemini-model" className="control-label">Model</label>
              <select
                id="gemini-model"
                value={geminiModelSelect}
                onChange={(e) => onGeminiModelSelectChange(e.target.value)}
                disabled={llmSaving || !llm.reachable}
              >
                <option value="__env__">From .env ({llm.env_gemini_model?.trim() || "GEMINI_MODEL"})</option>
                {(llm.models ?? []).map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
                <option value="__custom__">Custom…</option>
              </select>
              {geminiModelSelect === "__custom__" ? (
                <input
                  type="text"
                  value={geminiCustomModel}
                  onChange={(e) => onGeminiCustomModelChange(e.target.value)}
                  placeholder="e.g. gemini-2.5-pro"
                  disabled={llmSaving}
                  className="control-input"
                  aria-label="Custom Gemini model name"
                />
              ) : null}
              <button
                type="button"
                onClick={() => void onApplyGeminiModel()}
                disabled={llmSaving || geminiModelApplyDisabled}
              >
                Apply model
              </button>
            </div>
          ) : null}
          <p className="meta-line">
            {llm.provider === "gemini" ? (
              <>
                <code>{llm.configured_model || "—"}</code>
                {llm.gemini_model_ui_override ? (
                  <> · override (env <code>{llm.env_gemini_model || "—"}</code>)</>
                ) : null}
                {llm.ui_override ? (
                  <> · UI override (env: <code>{llm.env_provider ?? "ollama"}</code>)</>
                ) : null}
              </>
            ) : (
              <>
                <code>{llm.configured_host || "—"}</code> · <code>{llm.configured_model || "—"}</code>
                {llm.ollama_model_ui_override ? (
                  <> · override (env <code>{llm.env_ollama_model || "—"}</code>)</>
                ) : null}
              </>
            )}
            {llm.reachable && llm.model_installed === false ? (
              <span className="meta-warn">
                {" "}
                — {llm.provider === "gemini" ? "Check GEMINI_MODEL." : `Try: ollama pull ${llm.configured_model || "qwen3:4b-instruct"}`}
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
          <span className="badge">Setup</span>
        </div>
        <button type="button" onClick={onOpenSetupWizard}>Open setup wizard</button>
      </div>

      <div className="settings-block">
        <div className="settings-block-head">
          <span className="badge">Account</span>
          <span className={session?.signed_in ? "badge ok" : "badge"}>
            {session?.signed_in ? "Signed in" : "Not connected"}
          </span>
        </div>
        <div className="control-row">
          <a className="btn-link primary" href="/login">Connect Spotify</a>
          <button type="button" className="secondary" onClick={() => void onLogout()} disabled={!session?.signed_in}>
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
          <label htmlFor="device" className="control-label">Playback</label>
          <select
            id="device"
            value={deviceId}
            onChange={(e) => onDeviceIdChange(e.target.value)}
            disabled={!session?.signed_in || loadingDevices}
          >
            {deviceOptions.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
          <button type="button" className="secondary" onClick={() => void onRefreshDevices()} disabled={!session?.signed_in}>
            Refresh
          </button>
          <button type="button" onClick={() => void onSaveDevice()} disabled={!session?.signed_in}>
            Save
          </button>
        </div>
        <p className="hint tight">Open Spotify on this machine so a Connect device appears. Playback uses the saved device.</p>
      </div>
    </div>
  );
}
