import { useEffect, useState } from "react";

type Props = {
  /** When true, show black/green gradient instead of album art. */
  idle: boolean;
  artUrl: string | null;
};

export function LiquidBackground({ idle, artUrl }: Props) {
  const [activeUrl, setActiveUrl] = useState<string | null>(null);
  const [previousUrl, setPreviousUrl] = useState<string | null>(null);

  useEffect(() => {
    if (idle || !artUrl) {
      setActiveUrl(null);
      setPreviousUrl(null);
      return;
    }
    if (artUrl === activeUrl) return;
    setPreviousUrl(activeUrl);
    setActiveUrl(artUrl);
    const id = window.setTimeout(() => setPreviousUrl(null), 900);
    return () => window.clearTimeout(id);
  }, [artUrl, idle, activeUrl]);

  const mode = idle || !activeUrl ? "idle" : "art";

  return (
    <div className="liquid-bg" data-testid="liquid-background" data-mode={mode} aria-hidden="true">
      <div className="liquid-bg-idle" />
      {previousUrl ? (
        <div
          className="liquid-bg-layer liquid-bg-layer--out"
          style={{ backgroundImage: `url(${previousUrl})` }}
        />
      ) : null}
      {activeUrl && !idle ? (
        <div
          className="liquid-bg-layer liquid-bg-layer--in"
          style={{ backgroundImage: `url(${activeUrl})` }}
        />
      ) : null}
    </div>
  );
}
