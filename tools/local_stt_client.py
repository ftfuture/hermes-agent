"""Client for the loopback-only shared local STT sidecar."""

from __future__ import annotations

import base64
import ipaddress
import json
from pathlib import Path
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import build_opener, HTTPRedirectHandler, Request


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _error(message: str) -> dict[str, Any]:
    return {"success": False, "transcript": "", "error": message}


def transcribe_via_sidecar(
    file_path: str,
    model_name: str,
    *,
    url: str,
    token_file: Path | str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    parsed = urlparse(url)
    try:
        host = ipaddress.ip_address(parsed.hostname or "")
    except ValueError:
        host = None
    if parsed.scheme != "http" or host != ipaddress.IPv4Address("127.0.0.1"):
        return _error("Local STT sidecar URL must use literal IPv4 loopback 127.0.0.1")
    try:
        port = parsed.port
    except ValueError:
        port = None
    if (
        port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        return _error("Local STT sidecar URL must be a bare HTTP origin with a valid port")
    try:
        token = Path(token_file).read_text(encoding="utf-8").strip()
    except OSError as exc:
        return _error(f"Local STT sidecar token unavailable: {exc}")
    if not token:
        return _error("Local STT sidecar token is empty")

    path = Path(file_path)
    try:
        audio = path.read_bytes()
    except OSError as exc:
        return _error(f"Unable to read audio for local STT sidecar: {exc}")

    payload = json.dumps(
        {
            "audio_b64": base64.b64encode(audio).decode("ascii"),
            "suffix": path.suffix.lower() or ".wav",
            "model": model_name,
        }
    ).encode("utf-8")
    request = Request(
        url.rstrip("/") + "/v1/transcribe",
        data=payload,
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with build_opener(_NoRedirectHandler()).open(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("error", str(exc))
        except Exception:
            detail = str(exc)
        return _error(f"Local STT sidecar request failed ({exc.code}): {detail}")
    except (URLError, TimeoutError, socket.timeout, OSError) as exc:
        return _error(f"Local STT sidecar unavailable: {exc}")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return _error(f"Local STT sidecar returned invalid JSON: {exc}")

    if not isinstance(result, dict):
        return _error("Local STT sidecar returned invalid response")
    if result.get("success"):
        result["provider"] = "local_sidecar"
    return result
