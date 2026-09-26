"""Persist setup secrets (keyring when available, else chmod-0600 file in DATA_DIR).

Precedence for effective credentials (documented in README):
  1. Environment variables / backend `.env` (highest — existing deployments)
  2. OS keychain via `keyring` (when a usable backend exists)
  3. `secrets.json` in DATA_DIR (mode 0600 fallback)
  4. Non-secret fields in `setup.json` in DATA_DIR (e.g. Spotify client id, Ollama host)
"""

from __future__ import annotations

import json
import logging
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from spot_backend.config import Settings

logger = logging.getLogger(__name__)

_SERVICE_NAME = "spot-ai-fy"
_SETUP_FILE = "setup.json"
_SECRETS_FILE = "secrets.json"

_SECRET_KEYS = frozenset({"gemini_api_key"})

_PRIVATE_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR


def _setup_path(data_dir: Path) -> Path:
    return data_dir / _SETUP_FILE


def _secrets_path(data_dir: Path) -> Path:
    return data_dir / _SECRETS_FILE


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def _write_json_private(path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically with private mode from file creation (never world-readable)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        try:
            os.fchmod(fd, _PRIVATE_FILE_MODE)
        except (OSError, AttributeError):
            pass
        os.write(fd, payload)
        os.close(fd)
        fd = -1
        os.replace(tmp_name, str(path))
        tmp_name = ""
        try:
            os.chmod(path, _PRIVATE_FILE_MODE)
        except OSError:
            pass
    finally:
        if fd >= 0:
            os.close(fd)
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def read_setup_fields(data_dir: Path) -> dict[str, Any]:
    return _load_json(_setup_path(data_dir))


def write_setup_fields(data_dir: Path, patch: dict[str, Any]) -> None:
    from spot_backend.data_dir_lock import data_dir_lock

    with data_dir_lock(data_dir):
        cur = read_setup_fields(data_dir)
        for k, v in patch.items():
            if v is None:
                cur.pop(k, None)
            else:
                cur[k] = v
        data_dir.mkdir(parents=True, exist_ok=True)
        _write_json_private(_setup_path(data_dir), cur)


def _keyring_get(key: str) -> str | None:
    try:
        import keyring  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    try:
        val = keyring.get_password(_SERVICE_NAME, key)
    except Exception:  # noqa: BLE001 — keyring backends vary
        logger.debug("keyring get failed for %s", key, exc_info=True)
        return None
    if val is None:
        return None
    s = str(val).strip()
    return s or None


def _keyring_set(key: str, value: str) -> bool:
    try:
        import keyring  # pylint: disable=import-outside-toplevel
    except ImportError:
        return False
    try:
        keyring.set_password(_SERVICE_NAME, key, value)
        return True
    except Exception:  # noqa: BLE001
        logger.debug("keyring set failed for %s", key, exc_info=True)
        return False


def _keyring_delete(key: str) -> None:
    try:
        import keyring  # pylint: disable=import-outside-toplevel
    except ImportError:
        return
    try:
        keyring.delete_password(_SERVICE_NAME, key)
    except Exception:  # noqa: BLE001
        logger.debug("keyring delete failed for %s", key, exc_info=True)


def _file_secrets(data_dir: Path) -> dict[str, str]:
    raw = _load_json(_secrets_path(data_dir))
    out: dict[str, str] = {}
    for k, v in raw.items():
        if k in _SECRET_KEYS and isinstance(v, str) and v.strip():
            out[k] = v.strip()
    return out


def _write_file_secret(data_dir: Path, key: str, value: str) -> None:
    raw = _load_json(_secrets_path(data_dir))
    raw[key] = value
    _write_json_private(_secrets_path(data_dir), raw)


def read_secret(data_dir: Path, key: str) -> str:
    if key not in _SECRET_KEYS:
        return ""
    kr = _keyring_get(key)
    if kr:
        return kr
    return _file_secrets(data_dir).get(key, "")


def write_secret(data_dir: Path, key: str, value: str) -> str:
    """Store secret; returns storage backend used: keyring | file."""
    from spot_backend.data_dir_lock import data_dir_lock

    if key not in _SECRET_KEYS:
        raise ValueError(f"unknown secret key: {key}")
    v = value.strip()
    if not v:
        raise ValueError("secret value must be non-empty")
    with data_dir_lock(data_dir):
        return _write_secret_locked(data_dir, key, v)


def _write_secret_locked(data_dir: Path, key: str, value: str) -> str:
    if _keyring_set(key, value):
        # Remove file copy so we don't keep duplicates.
        raw = _load_json(_secrets_path(data_dir))
        if key in raw:
            raw.pop(key, None)
            if raw:
                _write_json_private(_secrets_path(data_dir), raw)
            elif _secrets_path(data_dir).is_file():
                _secrets_path(data_dir).unlink()
        return "keyring"
    _write_file_secret(data_dir, key, value)
    return "file"


def mask_secret(value: str) -> str:
    """Masked display for UI; never log or return the raw secret."""
    if not value.strip():
        return ""
    return "••••••••"


def merge_settings_from_store(base: Settings) -> Settings:
    """Apply DATA_DIR setup + secrets when env fields are empty."""
    setup = read_setup_fields(base.data_dir)
    updates: dict[str, Any] = {}

    if not (base.spotify_client_id or "").strip():
        cid = str(setup.get("spotify_client_id") or "").strip()
        if cid:
            updates["spotify_client_id"] = cid

    if not (base.gemini_api_key or "").strip():
        key = read_secret(base.data_dir, "gemini_api_key")
        if key:
            updates["gemini_api_key"] = key

    if not (os.environ.get("OLLAMA_HOST") or "").strip():
        host = str(setup.get("ollama_host") or "").strip()
        if host:
            updates["ollama_host"] = host.rstrip("/")

    return base.model_copy(update=updates) if updates else base


def gemini_key_configured(data_dir: Path, env_key: str) -> bool:
    return bool((env_key or "").strip() or read_secret(data_dir, "gemini_api_key"))


def spotify_client_configured(data_dir: Path, env_client_id: str) -> bool:
    if (env_client_id or "").strip():
        return True
    return bool(str(read_setup_fields(data_dir).get("spotify_client_id") or "").strip())
