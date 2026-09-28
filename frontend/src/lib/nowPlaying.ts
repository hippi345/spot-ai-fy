import { apiFetch, readJson } from "./api";

export type NowPlayingTrack = {
  id: string;
  uri?: string;
  name: string;
  artists: string[];
  album: string;
  art_url: string | null;
  duration_ms: number;
};

export type NowPlayingQueueItem = {
  uri?: string;
  name: string;
  artists: string[];
  art_url: string | null;
};

/** Drop only a leading queue row that duplicates the currently playing track URI. */
export function filterUpNextQueue(
  queue: NowPlayingQueueItem[],
  currentUri: string | null | undefined,
): NowPlayingQueueItem[] {
  if (!queue.length || !currentUri?.trim()) return queue;
  const first = queue[0]?.uri?.trim();
  if (first && first === currentUri.trim()) return queue.slice(1);
  return queue;
}

export type NowPlayingPayload = {
  is_playing: boolean;
  track: NowPlayingTrack | null;
  progress_ms?: number;
  device?: { name: string; type: string } | null;
  queue?: NowPlayingQueueItem[];
  fetched_at?: number;
  stale?: boolean;
};

export const MOCK_NOW_PLAYING_PLAYING: NowPlayingPayload = {
  is_playing: true,
  track: {
    id: "sample-track-aurora",
    name: "Neon Harbor Lights",
    artists: ["The Glass Foxes"],
    album: "Velvet Lanterns",
    art_url: "https://placehold.co/512x512/e10600/f5f5f5/png?text=Blinding",
    duration_ms: 240_000,
  },
  progress_ms: 82_000,
  device: { name: "Sample Desk Speaker", type: "Computer" },
  queue: [
    {
      name: "Midnight Relay",
      artists: ["Coral Static"],
      art_url: "https://placehold.co/32x32/16161c/9b9bac/png?text=Q",
    },
    {
      name: "Paper Moon Engine",
      artists: ["Velvet Lanterns"],
      art_url: "https://placehold.co/32x32/16161c/9b9bac/png?text=Q",
    },
  ],
  fetched_at: Date.now() / 1000,
};

function mockFromUrl(): boolean {
  if (typeof window === "undefined") return false;
  const params = new URLSearchParams(window.location.search);
  return params.get("mockNp") === "1";
}

export function nowPlayingUsesMock(): boolean {
  return (
    import.meta.env.VITE_MOCK_NOW_PLAYING === "true" ||
    import.meta.env.VITE_MOCK_NOW_PLAYING === "1" ||
    mockFromUrl()
  );
}

async function readNowPlayingResponse(res: Response): Promise<NowPlayingPayload | null> {
  if (res.status === 401) {
    return null;
  }
  return readJson<NowPlayingPayload>(res);
}

export async function fetchNowPlaying(): Promise<NowPlayingPayload | null> {
  if (nowPlayingUsesMock()) {
    return { ...MOCK_NOW_PLAYING_PLAYING, fetched_at: Date.now() / 1000 };
  }
  const res = await apiFetch("/api/now-playing");
  return readNowPlayingResponse(res);
}

export async function postPlayerToggle(): Promise<NowPlayingPayload | null> {
  if (nowPlayingUsesMock()) {
    const cur = MOCK_NOW_PLAYING_PLAYING;
    return {
      ...cur,
      is_playing: !cur.is_playing,
      fetched_at: Date.now() / 1000,
    };
  }
  const res = await apiFetch("/api/player/toggle", { method: "POST" });
  const data = await readNowPlayingResponse(res);
  if (data) return data;
  return fetchNowPlaying();
}

export async function postPlayerNext(): Promise<NowPlayingPayload | null> {
  if (nowPlayingUsesMock()) {
    return { ...MOCK_NOW_PLAYING_PLAYING, fetched_at: Date.now() / 1000 };
  }
  const res = await apiFetch("/api/player/next", { method: "POST" });
  const data = await readNowPlayingResponse(res);
  if (data) return data;
  return fetchNowPlaying();
}

export async function postPlayerPrevious(): Promise<NowPlayingPayload | null> {
  if (nowPlayingUsesMock()) {
    return { ...MOCK_NOW_PLAYING_PLAYING, fetched_at: Date.now() / 1000 };
  }
  const res = await apiFetch("/api/player/previous", { method: "POST" });
  const data = await readNowPlayingResponse(res);
  if (data) return data;
  return fetchNowPlaying();
}

/** Interpolate progress between server polls when playback is active. */
export function interpolateProgress(
  payload: NowPlayingPayload,
  nowMs: number,
): number {
  const base = payload.progress_ms ?? 0;
  const duration = payload.track?.duration_ms ?? 0;
  if (!payload.is_playing || !payload.track) {
    return base;
  }
  const fetchedSec = payload.fetched_at ?? 0;
  const elapsed = Math.max(0, nowMs - fetchedSec * 1000);
  const projected = base + elapsed;
  if (duration > 0) {
    return Math.min(projected, duration);
  }
  return projected;
}

export function pollIntervalMs(documentVisible: boolean): number {
  return documentVisible ? 3000 : 15000;
}
