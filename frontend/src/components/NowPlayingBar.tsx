import { useCallback, useEffect, useImperativeHandle, useRef, useState, forwardRef } from "react";

import {
  fetchNowPlaying,
  interpolateProgress,
  pollIntervalMs,
  postPlayerNext,
  postPlayerPrevious,
  postPlayerToggle,
  type NowPlayingPayload,
} from "../lib/nowPlaying";
import { NP_ART_PLACEHOLDER } from "../lib/spotifyImage";
import { IconNext, IconPause, IconPlay, IconPrevious } from "./icons/MediaControls";

export type NowPlayingBarHandle = {
  refresh: () => Promise<void>;
};

type Props = {
  signedIn: boolean;
  onBackgroundArtChange?: (info: {
    artUrl: string | null;
    hasTrack: boolean;
    isPlaying: boolean;
  }) => void;
};

function artistLine(artists: string[]): string {
  return artists.length ? artists.join(", ") : "Unknown artist";
}

function ArtImg({
  className,
  src,
  size,
}: {
  className: string;
  src: string | null | undefined;
  size: "main" | "queue";
}) {
  const [broken, setBroken] = useState(false);
  const resolved = !src || broken ? NP_ART_PLACEHOLDER : src;
  return (
    <img
      className={className}
      src={resolved}
      alt=""
      loading="lazy"
      decoding="async"
      onError={() => {
        if (!broken) setBroken(true);
      }}
      data-np-art={size}
    />
  );
}

export const NowPlayingBar = forwardRef<NowPlayingBarHandle, Props>(function NowPlayingBar(
  { signedIn, onBackgroundArtChange },
  ref,
) {
  const [payload, setPayload] = useState<NowPlayingPayload | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [tick, setTick] = useState(() => Date.now());
  const loadingRef = useRef(false);

  const refresh = useCallback(async () => {
    if (!signedIn) {
      setPayload(null);
      return;
    }
    if (loadingRef.current) return;
    loadingRef.current = true;
    try {
      const data = await fetchNowPlaying();
      if (data) {
        setPayload({ ...data, fetched_at: data.fetched_at ?? Date.now() / 1000 });
      } else {
        setPayload(null);
      }
    } catch {
      /* keep last payload on transient errors */
    } finally {
      loadingRef.current = false;
    }
  }, [signedIn]);

  useImperativeHandle(ref, () => ({ refresh }), [refresh]);

  useEffect(() => {
    if (!signedIn) {
      setPayload(null);
      return;
    }
    void refresh();
  }, [signedIn, refresh]);

  useEffect(() => {
    if (!signedIn) return undefined;
    let intervalId = 0;
    const schedule = () => {
      window.clearInterval(intervalId);
      const visible = document.visibilityState !== "hidden";
      intervalId = window.setInterval(() => {
        void refresh();
      }, pollIntervalMs(visible));
    };
    schedule();
    const onVis = () => {
      schedule();
      void refresh();
    };
    document.addEventListener("visibilitychange", onVis);
    return () => {
      document.removeEventListener("visibilitychange", onVis);
      window.clearInterval(intervalId);
    };
  }, [signedIn, refresh]);

  useEffect(() => {
    if (!payload?.is_playing) return undefined;
    const id = window.setInterval(() => setTick(Date.now()), 500);
    return () => window.clearInterval(id);
  }, [payload?.is_playing, payload?.fetched_at]);

  useEffect(() => {
    if (!signedIn) {
      onBackgroundArtChange?.({ artUrl: null, hasTrack: false, isPlaying: false });
      return;
    }
    const track = payload?.track ?? null;
    onBackgroundArtChange?.({
      artUrl: track?.art_url ?? null,
      hasTrack: Boolean(track),
      isPlaying: Boolean(payload?.is_playing),
    });
  }, [signedIn, payload?.track, payload?.is_playing, onBackgroundArtChange]);

  if (!signedIn) {
    return null;
  }

  const track = payload?.track ?? null;
  const idle = !track;
  const progress = payload ? interpolateProgress(payload, tick) : 0;
  const duration = track?.duration_ms ?? 0;
  const pct = duration > 0 ? Math.min(100, (progress / duration) * 100) : 0;
  const queue = payload?.queue ?? [];

  const toggleExpanded = () => setExpanded((v) => !v);

  const onBodyKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      toggleExpanded();
    }
  };

  const runControl = async (action: () => Promise<NowPlayingPayload | null>) => {
    try {
      const data = await action();
      if (data) {
        setPayload({ ...data, fetched_at: data.fetched_at ?? Date.now() / 1000 });
        setTick(Date.now());
      }
    } catch {
      /* silent */
    }
  };

  return (
    <div className={`np-bar-wrap${expanded ? " np-bar-wrap--expanded" : ""}`}>
      <div
        className={`np-bar${idle ? " np-bar--idle" : ""}`}
        role="button"
        tabIndex={0}
        aria-expanded={expanded}
        aria-label={idle ? "Nothing playing. Expand queue." : `Now playing ${track?.name}. Expand queue.`}
        onClick={(e) => {
          if ((e.target as HTMLElement).closest(".np-controls")) return;
          toggleExpanded();
        }}
        onKeyDown={onBodyKeyDown}
      >
        <div className="np-art" aria-hidden={idle}>
          {track ? (
            <ArtImg className="np-art-img" src={track.art_url} size="main" />
          ) : (
            <span className="np-art-placeholder" />
          )}
        </div>
        <div className="np-meta">
          <div className="np-title">{idle ? "Nothing playing" : track?.name}</div>
          <div className="np-subtitle">
            {idle ? "Start playback in Spotify or ask in chat" : artistLine(track?.artists ?? [])}
          </div>
          <div className="np-progress" aria-hidden={idle}>
            <div className="np-progress-fill" style={{ width: `${pct}%` }} />
          </div>
        </div>
        {payload?.device?.name ? <div className="np-device">{payload.device.name}</div> : null}
        <div className="np-controls">
          <button
            type="button"
            className="np-btn"
            aria-label="Previous track"
            onClick={() => void runControl(postPlayerPrevious)}
          >
            <IconPrevious />
          </button>
          <button
            type="button"
            className="np-btn np-btn--primary"
            aria-label={payload?.is_playing ? "Pause" : "Play"}
            onClick={() => void runControl(postPlayerToggle)}
          >
            {payload?.is_playing ? <IconPause /> : <IconPlay />}
          </button>
          <button
            type="button"
            className="np-btn"
            aria-label="Next track"
            onClick={() => void runControl(postPlayerNext)}
          >
            <IconNext />
          </button>
        </div>
      </div>
      {expanded ? (
        <ul className="np-queue" aria-label="Up next">
          {queue.length === 0 ? <li className="np-queue-empty">Queue is empty</li> : null}
          {queue.map((item, i) => (
            <li key={`${item.name}-${i}`} className="np-queue-item">
              <ArtImg className="np-queue-art" src={item.art_url} size="queue" />
              <span className="np-queue-text">
                <span className="np-queue-title">{item.name}</span>
                <span className="np-queue-artists">{artistLine(item.artists)}</span>
              </span>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
});
