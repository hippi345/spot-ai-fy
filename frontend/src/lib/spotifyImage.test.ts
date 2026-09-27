import { describe, expect, it } from "vitest";

import { pickSpotifyImageUrl } from "./spotifyImage";

describe("pickSpotifyImageUrl", () => {
  it("ignores empty urls and picks nearest size", () => {
    const url = pickSpotifyImageUrl(
      [
        { url: "", width: 640 },
        { url: "https://cdn.test/ok.jpg", width: 0, height: 64 },
        { url: "https://cdn.test/large.jpg", width: 640 },
      ],
      64,
    );
    expect(url).toBe("https://cdn.test/ok.jpg");
  });
});
