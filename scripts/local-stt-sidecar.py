#!/usr/bin/env python
"""Run the loopback-only shared faster-whisper sidecar."""

from __future__ import annotations

import argparse
from pathlib import Path
import threading

from tools import transcription_tools as stt
from tools.local_stt_sidecar import LocalSttSidecar, serve_in_thread


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--temp-dir", default="")
    parser.add_argument("--max-audio-mib", type=int, default=25)
    parser.add_argument("--idle-unload-seconds", type=float, default=600)
    args = parser.parse_args()

    token = Path(args.token_file).read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit("STT sidecar token file is empty")

    def state():
        return {
            "loaded": stt._local_model is not None,
            "model": stt._local_model_name,
        }

    app = LocalSttSidecar(
        token=token,
        transcribe_fn=stt._transcribe_local,
        unload_fn=stt.cleanup_local_stt,
        state_fn=state,
        temp_dir=Path(args.temp_dir) if args.temp_dir else None,
        max_audio_bytes=args.max_audio_mib * 1024 * 1024,
        idle_unload_seconds=args.idle_unload_seconds,
    )
    try:
        with serve_in_thread(app, host="127.0.0.1", port=args.port) as url:
            print(f"Shared local STT ready at {url}", flush=True)
            threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        stt.cleanup_local_stt()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
