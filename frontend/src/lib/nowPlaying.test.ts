import { describe, expect, it } from "vitest";

import { filterUpNextQueue } from "./nowPlaying";

describe("filterUpNextQueue", () => {
  const q = [
    { uri: "spotify:track:aaa", name: "A", artists: ["X"], art_url: null },
    { uri: "spotify:track:bbb", name: "B", artists: ["Y"], art_url: null },
  ];

  it("drops only a leading duplicate of the current URI", () => {
    expect(filterUpNextQueue(q, "spotify:track:aaa")).toEqual([q[1]]);
    expect(filterUpNextQueue(q, "spotify:track:bbb")).toEqual(q);
    expect(filterUpNextQueue(q, null)).toEqual(q);
  });
});
