/**
 * Typed HTTP helpers for the Spot-AI-fy backend.
 * Set VITE_API_BASE_URL to talk to a remote API; otherwise same-origin / Vite proxy.
 */

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

export function apiUrl(path: string): string {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return API_BASE ? `${API_BASE}${normalized}` : normalized;
}

export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  return fetch(apiUrl(path), init);
}

export async function readJson<T>(res: Response): Promise<T> {
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

export type SetupStatus = {
  spotify_configured: boolean;
  spotify_signed_in: boolean;
  llm_ready: boolean;
  provider: "ollama" | "gemini";
  redirect_uri: string;
  spotify_client_id_masked?: string;
  gemini_api_key_masked?: string;
  ollama_host?: string;
  ollama_model?: string;
  gemini_model?: string;
  llm_error?: string | null;
  setup_complete: boolean;
};

export async function fetchSetupStatus(): Promise<SetupStatus> {
  return readJson<SetupStatus>(await apiFetch("/api/setup/status"));
}

export async function saveSpotifyApp(clientId: string): Promise<{ redirect_uri: string }> {
  return readJson(
    await apiFetch("/api/setup/spotify-app", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ client_id: clientId }),
    }),
  );
}

export async function saveLlmSetup(body: {
  provider: "ollama" | "gemini";
  gemini_api_key?: string;
  ollama_host?: string;
  ollama_model?: string;
  gemini_model?: string;
  test?: boolean;
}): Promise<{ reachable?: boolean; models?: string[] }> {
  return readJson(
    await apiFetch("/api/setup/llm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

export async function probeOllama(host: string): Promise<{
  reachable: boolean;
  models: string[];
  error: string | null;
}> {
  const q = new URLSearchParams({ host });
  return readJson(await apiFetch(`/api/setup/ollama/probe?${q.toString()}`));
}
