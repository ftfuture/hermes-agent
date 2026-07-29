from pathlib import Path
from unittest.mock import patch

from tools import transcription_tools as tt


def _config(tmp_path, **local):
    return {
        "enabled": True,
        "provider": "local",
        "local": {"model": "model-path", **local},
    }


def test_local_sidecar_backend_returns_sidecar_result(tmp_path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"RIFF")
    config = _config(
        tmp_path,
        backend="sidecar",
        sidecar_url="http://127.0.0.1:8765",
        sidecar_token_file=str(tmp_path / "token"),
    )
    expected = {"success": True, "transcript": "공유", "provider": "local_sidecar"}
    with patch.object(tt, "_load_stt_config", return_value=config), patch(
        "tools.local_stt_client.transcribe_via_sidecar", return_value=expected
    ) as sidecar, patch.object(tt, "_transcribe_local") as local:
        result = tt.transcribe_audio(str(audio))
    assert result == expected
    sidecar.assert_called_once()
    local.assert_not_called()


def test_local_sidecar_failure_falls_back_when_enabled(tmp_path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"RIFF")
    config = _config(
        tmp_path,
        backend="sidecar",
        sidecar_url="http://127.0.0.1:8765",
        sidecar_token_file=str(tmp_path / "token"),
        fallback_to_in_process=True,
    )
    failed = {"success": False, "transcript": "", "error": "down"}
    fallback = {"success": True, "transcript": "로컬", "provider": "local"}
    with patch.object(tt, "_load_stt_config", return_value=config), patch(
        "tools.local_stt_client.transcribe_via_sidecar", return_value=failed
    ), patch.object(tt, "_transcribe_local", return_value=fallback) as local:
        result = tt.transcribe_audio(str(audio))
    assert result == fallback
    local.assert_called_once_with(str(audio), "model-path")


def test_local_sidecar_failure_is_returned_when_fallback_disabled(tmp_path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"RIFF")
    config = _config(
        tmp_path,
        backend="sidecar",
        sidecar_url="http://127.0.0.1:8765",
        sidecar_token_file=str(tmp_path / "token"),
        fallback_to_in_process=False,
    )
    failed = {"success": False, "transcript": "", "error": "down"}
    with patch.object(tt, "_load_stt_config", return_value=config), patch(
        "tools.local_stt_client.transcribe_via_sidecar", return_value=failed
    ), patch.object(tt, "_transcribe_local") as local:
        result = tt.transcribe_audio(str(audio))
    assert result == failed
    local.assert_not_called()


def test_local_sidecar_string_false_disables_fallback(tmp_path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"RIFF")
    config = _config(
        tmp_path,
        backend="sidecar",
        sidecar_token_file=str(tmp_path / "token"),
        fallback_to_in_process="false",
    )
    failed = {"success": False, "transcript": "", "error": "down"}
    with patch.object(tt, "_load_stt_config", return_value=config), patch(
        "tools.local_stt_client.transcribe_via_sidecar", return_value=failed
    ), patch.object(tt, "_transcribe_local") as local:
        result = tt.transcribe_audio(str(audio))
    assert result == failed
    local.assert_not_called()


def test_local_sidecar_string_true_enables_fallback(tmp_path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"RIFF")
    config = _config(
        tmp_path,
        backend="sidecar",
        sidecar_token_file=str(tmp_path / "token"),
        fallback_to_in_process="true",
    )
    failed = {"success": False, "transcript": "", "error": "down"}
    fallback = {"success": True, "transcript": "local", "provider": "local"}
    with patch.object(tt, "_load_stt_config", return_value=config), patch(
        "tools.local_stt_client.transcribe_via_sidecar", return_value=failed
    ), patch.object(tt, "_transcribe_local", return_value=fallback) as local:
        result = tt.transcribe_audio(str(audio))
    assert result == fallback
    local.assert_called_once()


def test_local_backend_defaults_to_in_process(tmp_path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"RIFF")
    config = _config(tmp_path)
    expected = {"success": True, "transcript": "기존", "provider": "local"}
    with patch.object(tt, "_load_stt_config", return_value=config), patch.object(
        tt, "_transcribe_local", return_value=expected
    ) as local:
        result = tt.transcribe_audio(str(audio))
    assert result == expected
    local.assert_called_once_with(str(audio), "model-path")
