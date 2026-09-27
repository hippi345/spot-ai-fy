export type SpotifyImage = { url?: string | null; width?: number | null; height?: number | null };

function sortKey(im: SpotifyImage): number {
  const w = im.width ?? 0;
  const h = im.height ?? 0;
  if (w > 0) return w;
  if (h > 0) return h;
  return 300;
}

export function pickSpotifyImageUrl(
  images: SpotifyImage[] | null | undefined,
  targetPx: number,
): string | null {
  if (!images?.length) return null;
  const valid = images.filter(
    (im) => typeof im.url === "string" && im.url.trim().startsWith("http"),
  );
  if (!valid.length) return null;
  valid.sort((a, b) => Math.abs(sortKey(a) - targetPx) - Math.abs(sortKey(b) - targetPx));
  return valid[0].url ?? null;
}

export const NP_ART_PLACEHOLDER =
  "data:image/svg+xml," +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" fill="#1a1a22"/><text x="50%" y="54%" text-anchor="middle" fill="#9b9bac" font-size="10" font-family="system-ui">♪</text></svg>',
  );
