import base64
import json
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from tools.local_stt_sidecar import LocalSttSidecar, serve_in_thread


def _request(url, token, path, payload=None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(
        url + path,
        data=body,
        method="GET" if payload is None else "POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=3) as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.mark.parametrize("host", ["localhost", "::1"])
def test_server_rejects_non_ipv4_literal_loopback_hosts(host):
    app = LocalSttSidecar(token="secret", transcribe_fn=lambda *_: {"success": True})
    with pytest.raises(ValueError, match="literal IPv4 loopback 127.0.0.1"):
        with serve_in_thread(app, host=host):
            pass


def test_sidecar_rejects_missing_or_wrong_token(tmp_path):
    app = LocalSttSidecar(token="secret", transcribe_fn=lambda *_: {"success": True})
    with serve_in_thread(app) as url:
        status, body = _request(url, "wrong", "/health")
    assert status == 401
    assert body["error"] == "unauthorized"


def test_sidecar_health_reports_load_state(tmp_path):
    app = LocalSttSidecar(token="secret", transcribe_fn=lambda *_: {"success": True}, state_fn=lambda: {"loaded": False})
    with serve_in_thread(app) as url:
        status, body = _request(url, "secret", "/health")
    assert status == 200
    assert body == {"ok": True, "loaded": False, "busy": False}


def test_sidecar_decodes_audio_to_private_tempfile_and_deletes_it(tmp_path):
    observed = {}

    def transcribe(path, model):
        p = Path(path)
        observed["exists_during"] = p.exists()
        observed["bytes"] = p.read_bytes()
        observed["model"] = model
        observed["path"] = p
        return {"success": True, "transcript": "안녕하세요", "provider": "local"}

    app = LocalSttSidecar(token="secret", transcribe_fn=transcribe, temp_dir=tmp_path)
    payload = {
        "audio_b64": base64.b64encode(b"RIFF-test-audio").decode("ascii"),
        "suffix": ".wav",
        "model": "model-path",
    }
    with serve_in_thread(app) as url:
        status, body = _request(url, "secret", "/v1/transcribe", payload)

    assert status == 200
    assert body["transcript"] == "안녕하세요"
    assert observed["exists_during"] is True
    assert observed["bytes"] == b"RIFF-test-audio"
    assert observed["model"] == "model-path"
    assert not observed["path"].exists()


def test_sidecar_rejects_oversized_base64_before_transcription(tmp_path):
    called = False

    def transcribe(*_):
        nonlocal called
        called = True

    app = LocalSttSidecar(token="secret", transcribe_fn=transcribe, max_audio_bytes=4, temp_dir=tmp_path)
    payload = {"audio_b64": base64.b64encode(b"12345").decode("ascii"), "suffix": ".wav", "model": "m"}
    with serve_in_thread(app) as url:
        status, body = _request(url, "secret", "/v1/transcribe", payload)
    assert status == 413
    assert body["error"] == "audio_too_large"
    assert called is False


def test_sidecar_serializes_concurrent_transcriptions(tmp_path):
    entered = []
    release = threading.Event()

    def transcribe(path, model):
        entered.append(time.monotonic())
        if len(entered) == 1:
            release.wait(timeout=2)
        return {"success": True, "transcript": model}

    app = LocalSttSidecar(token="secret", transcribe_fn=transcribe, temp_dir=tmp_path)
    payload = lambda model: {"audio_b64": base64.b64encode(b"x").decode("ascii"), "suffix": ".wav", "model": model}
    results = []
    with serve_in_thread(app) as url:
        first = threading.Thread(target=lambda: results.append(_request(url, "secret", "/v1/transcribe", payload("one"))))
        second = threading.Thread(target=lambda: results.append(_request(url, "secret", "/v1/transcribe", payload("two"))))
        first.start()
        while not entered:
            time.sleep(0.01)
        second.start()
        time.sleep(0.1)
        assert len(entered) == 1
        release.set()
        first.join(timeout=3)
        second.join(timeout=3)
    assert len(entered) == 2
    assert all(status == 200 for status, _ in results)


def test_unload_returns_busy_during_inference_and_runs_when_idle(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    unloaded = []

    def transcribe(path, model):
        entered.set()
        release.wait(timeout=2)
        return {"success": True, "transcript": "ok"}

    app = LocalSttSidecar(token="secret", transcribe_fn=transcribe, unload_fn=lambda: unloaded.append(True), temp_dir=tmp_path)
    payload = {"audio_b64": base64.b64encode(b"x").decode("ascii"), "suffix": ".wav", "model": "m"}
    with serve_in_thread(app) as url:
        worker = threading.Thread(target=lambda: _request(url, "secret", "/v1/transcribe", payload))
        worker.start()
        assert entered.wait(timeout=2)
        status, body = _request(url, "secret", "/v1/unload", {})
        assert status == 409
        assert body["error"] == "busy"
        release.set()
        worker.join(timeout=3)
        status, body = _request(url, "secret", "/v1/unload", {})
    assert status == 200
    assert body["unloaded"] is True
    assert unloaded == [True]


def test_idle_watchdog_unloads_once_after_deadline(tmp_path):
    now = [100.0]
    unloaded = []
    app = LocalSttSidecar(
        token="secret",
        transcribe_fn=lambda *_: {"success": True},
        unload_fn=lambda: unloaded.append(True),
        state_fn=lambda: {"loaded": True},
        temp_dir=tmp_path,
        idle_unload_seconds=10,
        clock=lambda: now[0],
    )
    app.mark_activity()
    now[0] = 109.9
    assert app.unload_if_idle() is False
    now[0] = 110.0
    assert app.unload_if_idle() is True
    assert app.unload_if_idle() is False
    assert unloaded == [True]


def test_idle_watchdog_rechecks_activity_after_acquiring_inference_lane(tmp_path):
    now = [100.0]
    unloaded = []
    app = LocalSttSidecar(
        token="secret",
        transcribe_fn=lambda *_: {"success": True},
        unload_fn=lambda: unloaded.append(True),
        state_fn=lambda: {"loaded": True},
        temp_dir=tmp_path,
        idle_unload_seconds=10,
        clock=lambda: now[0],
    )
    app.mark_activity()
    now[0] = 110.0
    real_lock = app._inference_lock

    class ActivityBeforeAcquire:
        def acquire(self, blocking=True):
            now[0] = 110.1
            app.mark_activity()
            return real_lock.acquire(blocking=blocking)

        def release(self):
            real_lock.release()

    app._inference_lock = ActivityBeforeAcquire()
    assert app.unload_if_idle() is False
    assert unloaded == []


def test_uncreatable_temp_directory_returns_structured_failure(tmp_path):
    unavailable = tmp_path / "not-a-directory"
    unavailable.write_text("occupied", encoding="utf-8")
    app = LocalSttSidecar(
        token="secret",
        transcribe_fn=lambda *_: {"success": True},
        temp_dir=unavailable,
    )
    payload = {
        "audio_b64": base64.b64encode(b"x").decode("ascii"),
        "suffix": ".wav",
        "model": "m",
    }
    status, body = app.transcribe(payload)
    assert status == 500
    assert body["success"] is False
    assert body["error"].startswith("transcription_failed:")
