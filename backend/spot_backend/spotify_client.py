from __future__ import annotations

import base64
import json
import time
from typing import Any

import httpx

from spot_backend.config import Settings, get_settings
from spot_backend.token_store import TokenBundle, is_expired, load_tokens, save_tokens


ACCOUNTS = "https://accounts.spotify.com"
API = "https://api.spotify.com/v1"


def _parse_json_or_none(r: httpx.Response) -> Any:
    """Return parsed JSON body, or None when the body is empty / non-JSON.

    Some Spotify player endpoints (e.g. PUT /me/player/repeat, /shuffle, /volume)
    return 200 OK with a non-JSON tracker token like ``-x6Q7rWK…`` instead of the
    documented 204 No Content. Those endpoints don't carry useful data, so we swallow
    JSON parse errors and treat the response as a successful no-op.
    """
    if not r.text:
        return None
    try:
        return r.json()
    except (json.JSONDecodeError, ValueError):
        return None

DEFAULT_SCOPES = (
    "user-read-private user-read-email "
    "playlist-read-private playlist-read-collaborative "
    "playlist-modify-public playlist-modify-private "
    "user-read-playback-state user-modify-playback-state user-library-read "
    "user-top-read user-follow-read"
)


def _normalize_token_scope_field(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, list):
        parts = [str(x).strip() for x in raw if x is not None and str(x).strip()]
        return " ".join(parts)
    return str(raw).strip()


class SpotifyAuthError(RuntimeError):
    pass


class SpotifyClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._http = httpx.Client(timeout=30.0)

    def close(self) -> None:
        self._http.close()

    def load_bundle(self) -> TokenBundle | None:
        return load_tokens(self.settings.resolved_token_path)

    def get_token_scopes(self) -> set[str]:
        """Scopes Spotify actually granted at consent (stale if consent predates a DEFAULT_SCOPES change)."""
        b = self.load_bundle()
        if not b or not b.scope:
            return set()
        return {s.strip() for s in b.scope.split() if s.strip()}

    def ensure_fresh_access_token(self) -> str:
        bundle = self.load_bundle()
        if not bundle or not bundle.access_token:
            raise SpotifyAuthError("Not signed in. Open the web UI and connect Spotify.")
        if bundle.refresh_token and is_expired(bundle):
            self._refresh(bundle)
            bundle = self.load_bundle()
        if not bundle:
            raise SpotifyAuthError("Token refresh failed.")
        return bundle.access_token

    def _refresh(self, bundle: TokenBundle) -> None:
        if not bundle.refresh_token:
            raise SpotifyAuthError("Missing refresh token; sign in again.")
        cid = self.settings.spotify_client_id
        if not cid:
            raise SpotifyAuthError("SPOTIFY_CLIENT_ID must be set.")
        secret = (self.settings.spotify_client_secret or "").strip()
        body: dict[str, str] = {
            "grant_type": "refresh_token",
            "refresh_token": bundle.refresh_token,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        if secret:
            auth = base64.b64encode(f"{cid}:{secret}".encode()).decode()
            headers["Authorization"] = f"Basic {auth}"
        else:
            body["client_id"] = cid
        r = self._http.post(f"{ACCOUNTS}/api/token", data=body, headers=headers)
        r.raise_for_status()
        data = r.json()
        new_access = data["access_token"]
        new_refresh = data.get("refresh_token") or bundle.refresh_token
        expires_in = int(data.get("expires_in", 3600))
        if "scope" in data and data.get("scope") is not None:
            scope_new = _normalize_token_scope_field(data.get("scope"))
            scope_out = scope_new if scope_new else bundle.scope
        else:
            scope_out = bundle.scope
        updated = TokenBundle(
            access_token=new_access,
            refresh_token=new_refresh,
            expires_at=time.time() + expires_in,
            scope=scope_out,
        )
        save_tokens(self.settings.resolved_token_path, updated)

    def exchange_authorization_code(
        self, code: str, redirect_uri: str, *, code_verifier: str
    ) -> TokenBundle:
        """PKCE public-client exchange (no client secret)."""
        cid = self.settings.spotify_client_id
        if not cid:
            raise SpotifyAuthError("SPOTIFY_CLIENT_ID must be set.")
        body = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": cid,
            "code_verifier": code_verifier,
        }
        r = self._http.post(
            f"{ACCOUNTS}/api/token",
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        r.raise_for_status()
        data = r.json()
        expires_in = int(data.get("expires_in", 3600))
        raw_scope = data.get("scope")
        if raw_scope is None:
            scope_str = DEFAULT_SCOPES.strip()
        else:
            scope_str = _normalize_token_scope_field(raw_scope)
            if not scope_str:
                scope_str = DEFAULT_SCOPES.strip()
        bundle = TokenBundle(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", ""),
            expires_at=time.time() + expires_in,
            scope=scope_str,
        )
        save_tokens(self.settings.resolved_token_path, bundle)
        return bundle

    def api_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        token = self.ensure_fresh_access_token()
        url = path if path.startswith("http") else f"{API}{path}"
        r = self._http.get(url, params=params, headers={"Authorization": f"Bearer {token}"})
        if r.status_code == 401:
            b = self.load_bundle()
            if b and b.refresh_token:
                self._refresh(b)
            token = self.ensure_fresh_access_token()
            r = self._http.get(url, params=params, headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        return _parse_json_or_none(r)

    def api_put(
        self,
        path: str,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        token = self.ensure_fresh_access_token()
        url = path if path.startswith("http") else f"{API}{path}"
        headers = {"Authorization": f"Bearer {token}"}
        kw: dict[str, Any] = {"headers": headers}
        if params is not None:
            kw["params"] = params
        if json_body is not None:
            kw["json"] = json_body
        r = self._http.put(url, **kw)
        if r.status_code == 401:
            b = self.load_bundle()
            if b and b.refresh_token:
                self._refresh(b)
            token = self.ensure_fresh_access_token()
            headers = {"Authorization": f"Bearer {token}"}
            kw = {"headers": headers}
            if params is not None:
                kw["params"] = params
            if json_body is not None:
                kw["json"] = json_body
            r = self._http.put(url, **kw)
        r.raise_for_status()
        return _parse_json_or_none(r)

    def api_post(self, path: str, json_body: dict[str, Any] | None = None, params: dict[str, Any] | None = None) -> Any:
        token = self.ensure_fresh_access_token()
        url = path if path.startswith("http") else f"{API}{path}"
        r = self._http.post(
            url, json=json_body, params=params, headers={"Authorization": f"Bearer {token}"}
        )
        if r.status_code == 401:
            b = self.load_bundle()
            if b and b.refresh_token:
                self._refresh(b)
            token = self.ensure_fresh_access_token()
            r = self._http.post(
                url, json=json_body, params=params, headers={"Authorization": f"Bearer {token}"}
            )
        r.raise_for_status()
        return _parse_json_or_none(r)

    def api_delete(
        self, path: str, json_body: dict[str, Any] | None = None, params: dict[str, Any] | None = None
    ) -> Any:
        # Use .request("DELETE", ...) rather than .delete(...) because httpx >= 0.28 removed
        # body kwargs (json=/data=/content=) from the convenience .delete() method. Spotify's
        # DELETE /playlists/{id}/items requires a JSON body, so we must send it via request().
        token = self.ensure_fresh_access_token()
        url = path if path.startswith("http") else f"{API}{path}"
        headers = {"Authorization": f"Bearer {token}"}
        kw: dict[str, Any] = {"headers": headers}
        if params is not None:
            kw["params"] = params
        if json_body is not None:
            kw["json"] = json_body
        r = self._http.request("DELETE", url, **kw)
        if r.status_code == 401:
            b = self.load_bundle()
            if b and b.refresh_token:
                self._refresh(b)
            token = self.ensure_fresh_access_token()
            headers = {"Authorization": f"Bearer {token}"}
            kw = {"headers": headers}
            if params is not None:
                kw["params"] = params
            if json_body is not None:
                kw["json"] = json_body
            r = self._http.request("DELETE", url, **kw)
        r.raise_for_status()
        return _parse_json_or_none(r)
