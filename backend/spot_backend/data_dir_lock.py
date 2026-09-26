"""Thread + cross-process locks for DATA_DIR read-merge-write."""

from __future__ import annotations

import threading
from pathlib import Path

from filelock import FileLock

_thread_guard = threading.Lock()
_locks: dict[str, FileLock] = {}


def data_dir_lock(data_dir: Path) -> FileLock:
    key = str(data_dir.resolve())
    with _thread_guard:
        lock = _locks.get(key)
        if lock is None:
            data_dir.mkdir(parents=True, exist_ok=True)
            lock = FileLock(str(data_dir / ".spot_ai_fy_setup.lock"))
            _locks[key] = lock
        return lock
