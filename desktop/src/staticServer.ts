import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { URL } from "node:url";

import type { IncomingMessage, ServerResponse } from "node:http";

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
};

function send(res: ServerResponse, status: number, body: string | Buffer, type?: string): void {
  res.statusCode = status;
  if (type) res.setHeader("Content-Type", type);
  res.end(body);
}

async function proxyToBackend(
  req: IncomingMessage,
  res: ServerResponse,
  backendOrigin: string,
): Promise<void> {
  const url = new URL(req.url ?? "/", backendOrigin);
  const headers: Record<string, string> = {};
  for (const [k, v] of Object.entries(req.headers)) {
    if (typeof v === "string") headers[k] = v;
  }
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.from(chunk));
  }
  const body = Buffer.concat(chunks);
  const upstream = await fetch(url, {
    method: req.method,
    headers,
    body: body.length ? body : undefined,
    redirect: "manual",
  });
  res.statusCode = upstream.status;
  upstream.headers.forEach((value, key) => {
    if (key.toLowerCase() === "transfer-encoding") return;
    res.setHeader(key, value);
  });
  const buf = Buffer.from(await upstream.arrayBuffer());
  res.end(buf);
}

export async function startUiServer(
  distDir: string,
  backendOrigin: string,
  preferredPort = 0,
): Promise<{ port: number; close: () => Promise<void> }> {
  const root = path.resolve(distDir);
  const server = http.createServer((req, res) => {
    void (async () => {
      const pathname = (req.url ?? "/").split("?")[0] ?? "/";
      if (
        pathname.startsWith("/api/") ||
        pathname === "/login" ||
        pathname === "/callback" ||
        pathname === "/logout"
      ) {
        await proxyToBackend(req, res, backendOrigin);
        return;
      }
      let filePath = path.join(root, pathname === "/" ? "index.html" : pathname);
      if (!filePath.startsWith(root)) {
        send(res, 403, "Forbidden");
        return;
      }
      if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
        filePath = path.join(root, "index.html");
      }
      const ext = path.extname(filePath);
      const type = MIME[ext] ?? "application/octet-stream";
      send(res, 200, fs.readFileSync(filePath), type);
    })().catch(() => {
      send(res, 500, "Internal error");
    });
  });
  await new Promise<void>((resolve) => server.listen(preferredPort, "127.0.0.1", () => resolve()));
  const addr = server.address();
  const port = typeof addr === "object" && addr ? addr.port : 0;
  return {
    port,
    close: () =>
      new Promise((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      }),
  };
}
