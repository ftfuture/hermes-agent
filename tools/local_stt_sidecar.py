"""Loopback-only HTTP sidecar for sharing a local STT model across gateways.

The transport accepts bounded base64 audio instead of filesystem paths so a
client cannot ask the service to read arbitrary local files.  Authentication
is mandatory even though the server binds only to loopback.
"""

from __future__ import annotations

import base64
import binascii
from contextlib import contextmanager
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Callable, Iterator, Optional

_ALLOWED_SUFFIXES = {".wav", ".mp3", ".ogg", ".opus", ".m4a", ".flac", ".webm"}


class LocalSttSidecar:
    """Request coordinator that owns one serialized STT execution lane."""

    def __init__(
        self,
        *,
        token: str,
        transcribe_fn: Callable[[str, str], dict[str, Any]],
        unload_fn: Optional[Callable[[], Any]] = None,
        state_fn: Optional[Callable[[], dict[str, Any]]] = None,
        temp_dir: Optional[Path] = None,
        max_audio_bytes: int = 25 * 1024 * 1024,
        idle_unload_seconds: float = 0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not token:
            raise ValueError("sidecar token must not be empty")
        self.token = token
        self.transcribe_fn = transcribe_fn
        self.unload_fn = unload_fn or (lambda: None)
        self.state_fn = state_fn or (lambda: {})
        self.temp_dir = Path(temp_dir) if temp_dir is not None else None
        self.max_audio_bytes = int(max_audio_bytes)
        self.idle_unload_seconds = max(float(idle_unload_seconds), 0.0)
        self._clock = clock
        self._inference_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._busy = False
        self._last_activity: Optional[float] = None
        self._unloaded_since_activity = False

    def mark_activity(self) -> None:
        with self._state_lock:
            self._last_activity = self._clock()
            self._unloaded_since_activity = False

    def unload_if_idle(self) -> bool:
        if self.idle_unload_seconds <= 0:
            return False
        with self._state_lock:
            if (
                self._busy
                or self._last_activity is None
                or self._unloaded_since_activity
                or self._clock() - self._last_activity < self.idle_unload_seconds
            ):
                return False
        if not self._inference_lock.acquire(blocking=False):
            return False
        try:
            # Activity may have completed after the optimistic check above but
            # before this lane was acquired.  Revalidate while the inference
            # lane is exclusively held so a fresh transcription is not evicted.
            with self._state_lock:
                if (
                    self._busy
                    or self._last_activity is None
                    or self._unloaded_since_activity
                    or self._clock() - self._last_activity < self.idle_unload_seconds
                ):
                    return False
            if not bool((self.state_fn() or {}).get("loaded")):
                with self._state_lock:
                    self._unloaded_since_activity = True
                return False
            self.unload_fn()
            with self._state_lock:
                self._unloaded_since_activity = True
            return True
        finally:
            self._inference_lock.release()

    def authorized(self, header: str) -> bool:
        prefix = "Bearer "
        if not header.startswith(prefix):
            return False
        return hmac.compare_digest(header[len(prefix) :], self.token)

    def health(self) -> dict[str, Any]:
        state = dict(self.state_fn() or {})
        with self._state_lock:
            busy = self._busy
        return {"ok": True, **state, "busy": busy}

    def transcribe(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        encoded = payload.get("audio_b64")
        model = payload.get("model")
        suffix = str(payload.get("suffix") or ".wav").lower()
        if not isinstance(encoded, str) or not isinstance(model, str) or not model:
            return 400, {"error": "invalid_request"}
        if suffix not in _ALLOWED_SUFFIXES:
            return 400, {"error": "invalid_suffix"}
        if len(encoded) > ((self.max_audio_bytes + 2) // 3) * 4 + 4:
            return 413, {"error": "audio_too_large"}
        try:
            audio = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            return 400, {"error": "invalid_base64"}
        if len(audio) > self.max_audio_bytes:
            return 413, {"error": "audio_too_large"}

        path: Optional[Path] = None
        try:
            if self.temp_dir:
                self.temp_dir.mkdir(parents=True, exist_ok=True)
            with self._inference_lock:
                with self._state_lock:
                    self._busy = True
                fd, raw_path = tempfile.mkstemp(suffix=suffix, dir=str(self.temp_dir) if self.temp_dir else None)
                path = Path(raw_path)
                try:
                    import os
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(audio)
                    result = self.transcribe_fn(str(path), model)
                finally:
                    with self._state_lock:
                        self._busy = False
                    self.mark_activity()
            if not isinstance(result, dict):
                return 500, {"error": "invalid_transcriber_response"}
            return (200 if result.get("success") else 500), result
        except Exception as exc:
            with self._state_lock:
                self._busy = False
            return 500, {"success": False, "error": f"transcription_failed: {exc}"}
        finally:
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def unload(self) -> tuple[int, dict[str, Any]]:
        with self._state_lock:
            if self._busy:
                return 409, {"error": "busy"}
        if not self._inference_lock.acquire(blocking=False):
            return 409, {"error": "busy"}
        try:
            self.unload_fn()
            with self._state_lock:
                self._unloaded_since_activity = True
            return 200, {"unloaded": True}
        finally:
            self._inference_lock.release()


def _handler_for(app: LocalSttSidecar):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HermesLocalSTT/1"

        def log_message(self, _format: str, *args: object) -> None:
            return

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorize(self) -> bool:
            if app.authorized(self.headers.get("Authorization", "")):
                return True
            self._send(401, {"error": "unauthorized"})
            return False

        def _read_json(self) -> Optional[dict[str, Any]]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return None
            max_json = ((app.max_audio_bytes + 2) // 3) * 4 + 64 * 1024
            if length <= 0 or length > max_json:
                return None
            try:
                value = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None
            return value if isinstance(value, dict) else None

        def do_GET(self) -> None:
            if not self._authorize():
                return
            if self.path == "/health":
                self._send(200, app.health())
            else:
                self._send(404, {"error": "not_found"})

        def do_POST(self) -> None:
            if not self._authorize():
                return
            payload = self._read_json()
            if payload is None:
                self._send(400, {"error": "invalid_json"})
                return
            if self.path == "/v1/transcribe":
                self._send(*app.transcribe(payload))
            elif self.path == "/v1/unload":
                self._send(*app.unload())
            else:
                self._send(404, {"error": "not_found"})

    return Handler


@contextmanager
def serve_in_thread(app: LocalSttSidecar, host: str = "127.0.0.1", port: int = 0) -> Iterator[str]:
    """Run a sidecar server for tests or an embedding process."""
    try:
        bind_address = ipaddress.ip_address(host)
    except ValueError:
        bind_address = None
    if bind_address != ipaddress.IPv4Address("127.0.0.1"):
        raise ValueError("local STT sidecar requires literal IPv4 loopback 127.0.0.1")
    server = ThreadingHTTPServer((host, port), _handler_for(app))
    thread = threading.Thread(target=server.serve_forever, name="local-stt-sidecar", daemon=True)
    watchdog_stop = threading.Event()

    def watchdog() -> None:
        interval = min(max(app.idle_unload_seconds / 4, 1.0), 5.0) if app.idle_unload_seconds else 5.0
        while not watchdog_stop.wait(interval):
            app.unload_if_idle()

    watchdog_thread = threading.Thread(target=watchdog, name="local-stt-idle-watchdog", daemon=True)
    thread.start()
    watchdog_thread.start()
    try:
        address, bound_port = server.server_address[:2]
        yield f"http://{address}:{bound_port}"
    finally:
        watchdog_stop.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        watchdog_thread.join(timeout=3)
