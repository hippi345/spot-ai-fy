import { afterEach, describe, expect, it, vi } from "vitest";

import { apiUrl } from "./api";

describe("apiUrl", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("uses same-origin paths when VITE_API_BASE_URL is unset", () => {
    vi.stubEnv("VITE_API_BASE_URL", "");
    expect(apiUrl("/api/setup/status")).toBe("/api/setup/status");
  });

  it("prefixes configured API base without trailing slash", () => {
    vi.stubEnv("VITE_API_BASE_URL", "https://api.example.com/");
    expect(apiUrl("/api/setup/status")).toBe("https://api.example.com/api/setup/status");
  });
});
