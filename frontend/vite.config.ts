/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const electronBuild = process.env.ELECTRON_BUILD === "1" || process.env.ELECTRON_BUILD === "true";

export default defineConfig({
  base: electronBuild ? "./" : "/",
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    globals: false,
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      // Chat can run for several minutes (Ollama + many tool rounds); avoid proxy cutting the connection.
      "/api": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
        timeout: 900_000,
        proxyTimeout: 900_000,
      },
      "/login": "http://127.0.0.1:8765",
      "/logout": "http://127.0.0.1:8765",
    },
  },
});
