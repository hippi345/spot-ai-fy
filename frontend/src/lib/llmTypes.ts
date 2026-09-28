export type LlmProviderId = "ollama" | "gemini" | "openai" | "anthropic" | "xai";

export const LLM_PROVIDER_IDS: LlmProviderId[] = [
  "ollama",
  "gemini",
  "openai",
  "anthropic",
  "xai",
];

export type LlmStatus = {
  provider?: LlmProviderId | string;
  provider_display?: string;
  env_provider?: string;
  ui_override?: boolean;
  configured_host?: string;
  configured_model: string;
  env_ollama_model?: string;
  ollama_model_ui_override?: boolean;
  env_gemini_model?: string;
  gemini_model_ui_override?: string | boolean;
  env_openai_model?: string;
  openai_model_ui_override?: boolean;
  env_anthropic_model?: string;
  anthropic_model_ui_override?: boolean;
  env_xai_model?: string;
  xai_model_ui_override?: boolean;
  api_key_masked?: string;
  api_key_configured?: boolean;
  reachable: boolean;
  models: string[] | null;
  model_installed?: boolean;
  error: string | null;
};

export function normalizeLlmProvider(value: string | undefined): LlmProviderId {
  const p = (value || "ollama").toLowerCase();
  if (p === "gemini" || p === "openai" || p === "anthropic" || p === "xai") return p;
  return "ollama";
}

export function providerLabel(id: LlmProviderId): string {
  switch (id) {
    case "gemini":
      return "Gemini";
    case "openai":
      return "OpenAI";
    case "anthropic":
      return "Anthropic";
    case "xai":
      return "xAI";
    default:
      return "Ollama";
  }
}

export function providerUsesApiKey(id: LlmProviderId): boolean {
  return id !== "ollama";
}
