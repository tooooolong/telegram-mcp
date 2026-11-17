"""Resource-limited local transcription worker."""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout

MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_AUDIO_SECONDS = 300
MAX_OUTPUT_CHARS = 20_000
WALL_SECONDS = 120
_ADDRESS_SPACE_BYTES = 8 * 1024 * 1024 * 1024
_CPU_SECONDS = 120
_SAMPLE_RATE = 16_000


class AudioLimitExceeded(ValueError):
    pass


def _apply_limits() -> bool:
    if sys.platform != "linux":
        return False
    import resource

    try:
        for kind, target in (
            (resource.RLIMIT_AS, _ADDRESS_SPACE_BYTES),
            (resource.RLIMIT_CPU, _CPU_SECONDS),
            (resource.RLIMIT_CORE, 0),
        ):
            current = resource.getrlimit(kind)
            limit = min((target, *(value for value in current if value != resource.RLIM_INFINITY)))
            resource.setrlimit(kind, (limit, limit))
    except (OSError, ValueError):
        return False
    return True


def _container_format(data: bytes) -> str:
    if data.startswith(b"OggS"):
        return "ogg"
    if data[4:8] == b"ftyp":
        return "mov"
    if data.startswith(b"\x1aE\xdf\xa3"):
        return "matroska"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data.startswith(b"ID3") or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0):
        return "mp3"
    raise ValueError("Unsupported local transcription audio format.")


def _deny_external_media(url, flags, options):
    raise OSError("External media sources are disabled.")


def _collect_samples(frames, resample, audio) -> int:
    count = 0
    for frame in frames:
        for converted in resample(frame):
            end = count + converted.samples
            if end > len(audio):
                raise AudioLimitExceeded
            audio[count:end] = converted.to_ndarray().reshape(-1)
            count = end
    return count


def _decode(data: bytes):
    from itertools import chain

    import av
    import numpy as np

    resampler = av.AudioResampler(format="s16", layout="mono", rate=_SAMPLE_RATE)
    audio = np.empty(_SAMPLE_RATE * MAX_AUDIO_SECONDS, dtype=np.int16)
    with av.open(
        io.BytesIO(data),
        mode="r",
        format=_container_format(data),
        metadata_errors="ignore",
        io_open=_deny_external_media,
    ) as container:
        count = _collect_samples(
            chain(container.decode(audio=0), (None,)), resampler.resample, audio
        )
    if not count:
        raise ValueError("Recording has no audio.")
    return audio[:count].astype(np.float32) / 32768


def _transcribe(data: bytes, settings: list, language: str | None) -> dict:
    from faster_whisper import WhisperModel

    audio = _decode(data)
    name, device, compute_type, download_root = settings
    model = WhisperModel(
        name, device=device, compute_type=compute_type, download_root=download_root, cpu_threads=2
    )
    segments, info = model.transcribe(audio, language=language, vad_filter=True)
    parts = []
    remaining = MAX_OUTPUT_CHARS
    for segment in segments:
        if len(segment.text) > remaining:
            return {"status": "error", "error": "Whisper exceeded the transcript output limit."}
        if segment.text:
            parts.append(segment.text)
            remaining -= len(segment.text)
    text = "".join(parts).strip()
    if not text:
        return {"status": "error", "error": "Empty transcript from Whisper."}
    return {"status": "ok", "text": text, "lang": getattr(info, "language", None) or language}


def _main() -> int:
    if not _apply_limits():
        return 3
    data = sys.stdin.buffer.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        return 1
    settings, language = json.loads(sys.argv[1])
    try:
        with redirect_stdout(sys.stderr):
            result = _transcribe(data, settings, language)
    except AudioLimitExceeded:
        result = {"status": "error", "error": f"Audio exceeds {MAX_AUDIO_SECONDS} seconds."}
    except Exception:
        result = {
            "status": "error",
            "error": "Whisper could not decode the recording or load the configured model.",
        }
    sys.stdout.buffer.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
