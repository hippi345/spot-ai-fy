import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import net from "node:net";

import treeKill from "tree-kill";

import { backendDir, resolvePythonExecutable } from "./paths";

export type BackendHandle = {
  port: number;
  child: ChildProcessWithoutNullStreams;
  stop: () => Promise<void>;
};

export async function pickFreePort(preferred = 8765): Promise<number> {
  const tryPort = (port: number) =>
    new Promise<boolean>((resolve) => {
      const srv = net.createServer();
      srv.once("error", () => resolve(false));
      srv.once("listening", () => {
        srv.close(() => resolve(true));
      });
      srv.listen(port, "127.0.0.1");
    });
  if (await tryPort(preferred)) return preferred;
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.listen(0, "127.0.0.1", () => {
      const addr = srv.address();
      const port = typeof addr === "object" && addr ? addr.port : 0;
      srv.close(() => resolve(port));
    });
    srv.on("error", reject);
  });
}

async function waitForHealth(port: number, timeoutMs = 60_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  const url = `http://127.0.0.1:${port}/api/health`;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(url);
      if (res.ok) {
        const body = (await res.json()) as { status?: string };
        if (body.status === "ok") return;
      }
    } catch {
      /* retry */
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`Backend health check failed on port ${port}`);
}

export async function startBackend(
  frontendOrigin: string,
  port = 0,
): Promise<BackendHandle> {
  const resolvedPort = port > 0 ? port : await pickFreePort(8765);
  const python = resolvePythonExecutable();
  const cwd = backendDir();
  const redirectUri = `http://127.0.0.1:${resolvedPort}/callback`;
  const env = {
    ...process.env,
    API_HOST: "127.0.0.1",
    API_PORT: String(resolvedPort),
    FRONTEND_ORIGIN: frontendOrigin,
    SPOTIFY_REDIRECT_URI: redirectUri,
  };
  const child = spawn(python, ["-m", "spot_backend"], {
    cwd,
    env,
    stdio: "pipe",
    windowsHide: true,
  });
  child.stdout?.on("data", (chunk) => {
    process.stdout.write(`[backend] ${chunk}`);
  });
  child.stderr?.on("data", (chunk) => {
    process.stderr.write(`[backend] ${chunk}`);
  });
  await waitForHealth(resolvedPort);
  return {
    port: resolvedPort,
    child,
    stop: async () => {
      const pid = child.pid;
      if (!pid) return;
      await new Promise<void>((resolve) => {
        treeKill(pid, "SIGTERM", (err) => {
          if (err) treeKill(pid, "SIGKILL", () => resolve());
          else resolve();
        });
      });
    },
  };
}
