import type { ReactNode } from "react";

import {
  type LlmProviderId,
  type LlmStatus,
  normalizeLlmProvider,
  providerLabel,
  providerUsesApiKey,
} from "../lib/llmTypes";

type Session = {
  signed_in: boolean;
  spotify_playlist_write_ok?: boolean | null;
};

type DeviceOption = { value: string; label: string };

export type ModelSpotifySettingsProps = {
  llm: LlmStatus | null;
  llmPick: LlmProviderId;
  llmSaving: boolean;
  modelSelect: string;
  modelCustom: string;
  modelApplyDisabled: boolean;
  apiKeyDraft: string;
  apiKeySaving: boolean;
  session: Session | null;
  deviceId: string;
  deviceOptions: DeviceOption[];
  loadingDevices: boolean;
  onLlmPickChange: (v: LlmProviderId) => void;
  onApplyLlmProvider: () => void;
  onResetLlmProvider: () => void;
  onRefreshLlm: () => void;
  onModelSelectChange: (v: string) => void;
  onModelCustomChange: (v: string) => void;
  onApplyModel: () => void;
  onApiKeyDraftChange: (v: string) => void;
  onSaveApiKey: () => void;
  onOpenSetupWizard: () => void;
  onLogout: () => void;
  onDeviceIdChange: (v: string) => void;
  onRefreshDevices: () => void;
  onSaveDevice: () => void;
};

function envModelLabel(llm: LlmStatus, provider: LlmProviderId): string {
  switch (provider) {
    case "gemini":
      return llm.env_gemini_model?.trim() || "GEMINI_MODEL";
    case "openai":
      return llm.env_openai_model?.trim() || "OPENAI_MODEL";
    case "anthropic":
      return llm.env_anthropic_model?.trim() || "ANTHROPIC_MODEL";
    case "xai":
      return llm.env_xai_model?.trim() || "XAI_MODEL";
    default:
      return llm.env_ollama_model?.trim() || "OLLAMA_MODEL";
  }
}

function modelOverrideActive(llm: LlmStatus, provider: LlmProviderId): boolean {
  switch (provider) {
    case "gemini":
      return Boolean(llm.gemini_model_ui_override);
    case "openai":
      return Boolean(llm.openai_model_ui_override);
    case "anthropic":
      return Boolean(llm.anthropic_model_ui_override);
    case "xai":
      return Boolean(llm.xai_model_ui_override);
    default:
      return Boolean(llm.ollama_model_ui_override);
  }
}

function llmModelSummary(llm: LlmStatus): { label: string; details: string } {
  const provider = normalizeLlmProvider(llm.provider);
  const model = (llm.configured_model || "").trim() || "—";
  const details: string[] = [`Backend: ${providerLabel(provider)}`, `Model: ${model}`];
  if (provider === "ollama" && llm.configured_host) {
    details.push(`Host: ${llm.configured_host}`);
  }
  if (llm.ui_override) {
    details.push(`UI provider override (env default: ${llm.env_provider ?? "ollama"})`);
  }
  if (modelOverrideActive(llm, provider)) {
    details.push(`Model override (env default: ${envModelLabel(llm, provider)})`);
  }
  return { label: `Using ${providerLabel(provider)} · ${model}`, details: details.join(" · ") };
}

export function ModelSpotifySettings(props: ModelSpotifySettingsProps): ReactNode {
  const {
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
    onLlmPickChange,
    onApplyLlmProvider,
    onResetLlmProvider,
    onRefreshLlm,
    onModelSelectChange,
    onModelCustomChange,
    onApplyModel,
    onApiKeyDraftChange,
    onSaveApiKey,
    onOpenSetupWizard,
    onLogout,
    onDeviceIdChange,
    onRefreshDevices,
    onSaveDevice,
  } = props;

  const activeProvider = llm ? normalizeLlmProvider(llm.provider) : llmPick;

  return (
    <div className="settings-body settings-body--sheet">
      {llm ? (
        <div className="settings-block settings-block--glass">
          <div className="settings-block-head">
            <span className="badge accent">{providerLabel(activeProvider)}</span>
            <span className={llm.reachable ? "badge ok" : "badge warn"}>{llm.reachable ? "OK" : "Issue"}</span>
          </div>
          <div className="control-row">
            <label htmlFor="llm-backend" className="control-label">Backend</label>
            <select
              id="llm-backend"
              className="glass-input"
              value={llmPick}
              onChange={(e) => onLlmPickChange(normalizeLlmProvider(e.target.value))}
              disabled={llmSaving}
            >
              <option value="ollama">Ollama (local)</option>
              <option value="gemini">Gemini</option>
              <option value="openai">OpenAI</option>
              <option value="anthropic">Anthropic (Claude)</option>
              <option value="xai">xAI (Grok)</option>
            </select>
            <button type="button" onClick={() => void onApplyLlmProvider()} disabled={llmSaving || llmPick === activeProvider}>
              Apply
            </button>
            <button type="button" className="secondary" onClick={() => void onResetLlmProvider()} disabled={llmSaving || !llm.ui_override}>
              Reset to .env
            </button>
            <button type="button" className="secondary" onClick={() => void onRefreshLlm()}>
              Refresh status
            </button>
          </div>

          {providerUsesApiKey(activeProvider) ? (
            <div className="control-row wrap settings-api-key-row">
              <label htmlFor="llm-api-key" className="control-label">API key</label>
              <input
                id="llm-api-key"
                type="password"
                className="control-input glass-input"
                value={apiKeyDraft}
                onChange={(e) => onApiKeyDraftChange(e.target.value)}
                placeholder={llm.api_key_masked ? `Saved (${llm.api_key_masked})` : "Paste API key"}
                autoComplete="off"
                disabled={apiKeySaving || llmSaving}
              />
              <button type="button" onClick={() => void onSaveApiKey()} disabled={apiKeySaving || llmSaving || !apiKeyDraft.trim()}>
                Save key
              </button>
            </div>
          ) : null}

          <div className="control-row wrap">
            <label htmlFor="llm-model" className="control-label">Model</label>
            <select
              id="llm-model"
              className="glass-input"
              value={modelSelect}
              onChange={(e) => onModelSelectChange(e.target.value)}
              disabled={llmSaving || (providerUsesApiKey(activeProvider) && !llm.reachable && !llm.api_key_configured)}
            >
              <option value="__env__">From .env ({envModelLabel(llm, activeProvider)})</option>
              {(llm.models ?? []).map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
              <option value="__custom__">Custom…</option>
            </select>
            {modelSelect === "__custom__" ? (
              <input
                type="text"
                value={modelCustom}
                onChange={(e) => onModelCustomChange(e.target.value)}
                disabled={llmSaving}
                className="control-input glass-input"
                aria-label="Custom model name"
              />
            ) : null}
            <button type="button" onClick={() => void onApplyModel()} disabled={llmSaving || modelApplyDisabled}>
              Apply model
            </button>
          </div>

          <p className="meta-line">
            <span className="meta-line-short" title={llmModelSummary(llm).details}>
              {llmModelSummary(llm).label}
            </span>
            {llm.reachable && llm.model_installed === false ? (
              <span className="meta-warn">
                {" "}
                — Check the model name or pull/install the model for {providerLabel(activeProvider)}.
              </span>
            ) : null}
          </p>
          {!llm.reachable ? (
            <p className="error tight">
              {llm.error ?? "Unreachable"}
              {activeProvider === "ollama"
                ? " Start Ollama or set OLLAMA_HOST."
                : " Set the API key in Settings or backend/.env."}
            </p>
          ) : null}
        </div>
      ) : null}

      <div className="settings-block settings-block--glass">
        <div className="settings-block-head">
          <span className="badge">Setup</span>
        </div>
        <button type="button" onClick={onOpenSetupWizard}>Open setup wizard</button>
      </div>

      <div className="settings-block settings-block--glass">
        <div className="settings-block-head">
          <span className="badge">Account</span>
          <span className={session?.signed_in ? "badge ok" : "badge"}>
            {session?.signed_in ? "Signed in" : "Not connected"}
          </span>
        </div>
        <div className="control-row">
          <a
            className={session?.signed_in ? "btn-link secondary ghost" : "btn-link primary"}
            href="/login"
          >
            {session?.signed_in ? "Reconnect" : "Connect Spotify"}
          </a>
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

      <div className="settings-block settings-block--glass">
        <div className="settings-block-head">
          <span className="badge">Device</span>
        </div>
        <div className="control-row wrap">
          <label htmlFor="device" className="control-label">Playback</label>
          <select
            id="device"
            className="glass-input"
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
