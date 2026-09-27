# Spot-AI-fy desktop (Electron)

The desktop app wraps the same React UI and FastAPI backend in an Electron shell so you are not tied to a browser tab.

## Prerequisites

- **Node.js 20+** (same as the web frontend)
- **Python 3.12+** with backend dependencies installed (`pip install -r backend/requirements.txt`)
- A **Spotify Developer** app with redirect URIs that match your backend port (default `http://127.0.0.1:8765/callback`; the desktop app may pick another free port — add `http://127.0.0.1:<port>/callback` in the Spotify dashboard if you use a non-default port)
- **Windows**: NSIS installer and portable builds are the primary packaging targets; window controls (min / max / close) appear in the custom title bar.

## Development

1. Build the web UI for Electron (relative asset paths):

   ```bash
   cd frontend
   ELECTRON_BUILD=1 npm run build
   ```

2. Install and run the desktop shell:

   ```bash
   cd ../desktop
   npm ci
   npm run dev
   ```

   `npm run dev` compiles the main process TypeScript and launches Electron. The main process:

   - Picks free local ports for the API and UI static server
   - Starts `python -m spot_backend` (prefers `backend/.venv` when present)
   - Waits for `GET /api/health`
   - Loads the built `frontend/dist` via a local HTTP server (proxies `/api`, `/login`, `/callback`, `/logout` to the backend)

3. **Spotify OAuth** opens in the **system browser** via the Connect link; the callback hits the loopback backend URL, then redirects back into the app.

Environment overrides:

| Variable | Purpose |
|----------|---------|
| `SPOT_AI_FY_PYTHON` | Path to Python executable |
| `SPOT_AI_FY_SMOKE=1` | Headless smoke mode (quit shortly after window load) |

## Production build (Windows)

From `desktop/` after `frontend` is built with `ELECTRON_BUILD=1`:

```bash
npm run dist
```

Artifacts land in `desktop/release/` (NSIS installer + portable on Windows).

### Bundling Python (stretch goal)

The installer currently expects **Python on the machine** (or a project venv). A bundled backend is not shipped yet. A practical approach for a follow-up:

1. **PyInstaller** one-file or one-folder spec for `spot_backend` + uvicorn
2. Ship the artifact under `resources/backend/` and spawn it instead of `python -m spot_backend`
3. Keep the same health-check and port env vars (`API_PORT`, `FRONTEND_ORIGIN`, `SPOTIFY_REDIRECT_URI`)

Document the chosen PyInstaller entrypoint in this file when implemented.

## Security notes

- `contextIsolation: true`, `nodeIntegration: false`, minimal `contextBridge` preload
- Content-Security-Policy applied to the renderer session
- API traffic stays on `127.0.0.1`

## Troubleshooting (Windows)

- If the window is blank, confirm `frontend/dist` exists (`ELECTRON_BUILD=1 npm run build`).
- If OAuth fails, verify the Spotify redirect URI includes your backend port (`http://127.0.0.1:<port>/callback`).
- Firewall prompts: allow loopback for the app; no inbound internet exposure is required for the API.
