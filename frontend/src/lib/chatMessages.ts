export const FRIENDLY_SPOTIFY_GUIDANCE =
  "I can help with Spotify things like playing music, searching, playlists, and queue management — try something like “play my workout playlist”, “search for Taylor Swift”, or “what’s playing?”";

export function isUnpersistedAssistantFallback(text: string): boolean {
  const t = text.trim();
  if (!t) return true;
  if (t === FRIENDLY_SPOTIFY_GUIDANCE) return true;
  if (t.startsWith("The model returned no assistant text")) return true;
  if (t.startsWith("No response from model.")) return true;
  return false;
}
