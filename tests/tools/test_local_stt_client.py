from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

from tools.local_stt_client import transcribe_via_sidecar
from tools.local_stt_sidecar import LocalSttSidecar, serve_in_thread


def test_client_sends_audio_and_returns_transcript(tmp_path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"RIFF-audio")
    token_file = tmp_path / "token"
    token_file.write_text("secret\n", encoding="utf-8")

    def transcribe(path, model):
        assert Path(path).read_bytes() == b"RIFF-audio"
        assert model == "whisper-model"
        return {"success": True, "transcript": "테스트", "provider": "local"}

    app = LocalSttSidecar(token="secret", transcribe_fn=transcribe, temp_dir=tmp_path / "server")
    with serve_in_thread(app) as url:
        result = transcribe_via_sidecar(str(audio), "whisper-model", url=url, token_file=token_file, timeout=3)
    assert result == {"success": True, "transcript": "테스트", "provider": "local_sidecar"}


def test_client_returns_structured_error_when_unavailable(tmp_path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"x")
    token_file = tmp_path / "token"
    token_file.write_text("secret", encoding="utf-8")
    result = transcribe_via_sidecar(str(audio), "m", url="http://127.0.0.1:1", token_file=token_file, timeout=0.1)
    assert result["success"] is False
    assert result["error"].startswith("Local STT sidecar unavailable:")


def test_client_rejects_empty_token_file(tmp_path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"x")
    token_file = tmp_path / "token"
    token_file.write_text("\n", encoding="utf-8")
    result = transcribe_via_sidecar(str(audio), "m", url="http://127.0.0.1:1", token_file=token_file)
    assert result == {"success": False, "transcript": "", "error": "Local STT sidecar token is empty"}


@pytest.mark.parametrize("url", ["http://localhost:8765", "http://[::1]:8765"])
def test_client_rejects_non_ipv4_literal_loopback_urls(tmp_path, url):
    result = transcribe_via_sidecar(
        str(tmp_path / "missing.wav"), "m", url=url, token_file=tmp_path / "missing-token"
    )
    assert result == {
        "success": False,
        "transcript": "",
        "error": "Local STT sidecar URL must use literal IPv4 loopback 127.0.0.1",
    }


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:bad",
        "http://user@127.0.0.1:8765",
        "http://127.0.0.1:8765/base",
        "http://127.0.0.1:8765?target=elsewhere",
        "http://127.0.0.1:8765#fragment",
    ],
)
def test_client_rejects_malformed_or_non_origin_loopback_urls(tmp_path, url):
    result = transcribe_via_sidecar(
        str(tmp_path / "missing.wav"), "m", url=url, token_file=tmp_path / "missing-token"
    )
    assert result == {
        "success": False,
        "transcript": "",
        "error": "Local STT sidecar URL must be a bare HTTP origin with a valid port",
    }


def test_client_rejects_redirect_without_forwarding_credentials_or_audio(tmp_path):
    received = []

    class Destination(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_args):
            pass

    destination = ThreadingHTTPServer(("127.0.0.1", 0), Destination)

    class Redirect(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{destination.server_port}/stolen")
            self.end_headers()

        def log_message(self, *_args):
            pass

    redirect = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in (destination, redirect)]
    for thread in threads:
        thread.start()
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"secret-audio")
    token_file = tmp_path / "token"
    token_file.write_text("secret-token", encoding="utf-8")
    try:
        result = transcribe_via_sidecar(
            str(audio), "m", url=f"http://127.0.0.1:{redirect.server_port}", token_file=token_file
        )
    finally:
        for server in (redirect, destination):
            server.shutdown()
            server.server_close()
    assert result["success"] is False
    assert "request failed (302)" in result["error"]
    assert received == []
