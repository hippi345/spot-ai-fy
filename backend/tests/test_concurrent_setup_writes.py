from __future__ import annotations

import threading

from spot_backend.secrets_store import read_setup_fields, write_setup_fields


def test_concurrent_setup_writes_merge_both_fields(data_dir) -> None:
    errors: list[Exception] = []

    def write_cid() -> None:
        try:
            write_setup_fields(data_dir, {"spotify_client_id": "a" * 32})
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def write_host() -> None:
        try:
            write_setup_fields(data_dir, {"ollama_host": "http://127.0.0.1:11434"})
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    t1 = threading.Thread(target=write_cid)
    t2 = threading.Thread(target=write_host)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert not errors
    merged = read_setup_fields(data_dir)
    assert merged.get("spotify_client_id") == "a" * 32
    assert merged.get("ollama_host") == "http://127.0.0.1:11434"
