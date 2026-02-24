import io
import os
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from app.core.config import settings
from kokoro import KPipeline

_PIPELINE: Optional[KPipeline] = None
_LANG_IN_USE: Optional[str] = None


def _get_pipeline(lang_code: str) -> KPipeline:
    global _PIPELINE, _LANG_IN_USE
    if _PIPELINE is None or _LANG_IN_USE != lang_code:
        _PIPELINE = KPipeline(lang_code=lang_code)
        _LANG_IN_USE = lang_code
    return _PIPELINE


def _as_float32_mono(x) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass

    return np.asarray(x, dtype=np.float32).reshape(-1)


def synthesize_np(
    text: str,
    voice: Optional[str] = None,
    speed: float = 1.0,
    lang_code: Optional[str] = None,
    sample_rate: Optional[int] = None,
) -> Tuple[np.ndarray, int]:

    voice = voice or settings.default_voice
    lang_code = lang_code or settings.lang_code
    sr = int(sample_rate or settings.default_sample_rate)

    pipe = _get_pipeline(lang_code=lang_code)

    voices = [v.strip() for v in (voice or "").split("+") if v.strip()]
    if not voices:
        voices = [settings.default_voice]

    rendered = []

    for v in voices:
        chunks = []
        gen = pipe(text, voice=v, speed=float(speed), split_pattern=r"\n+")

        for (_gs, _ps, audio) in gen:
            chunks.append(_as_float32_mono(audio))

        if chunks:
            rendered.append(
                np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            )

    if not rendered:
        return np.zeros(0, dtype=np.float32), sr

    if len(rendered) == 1:
        return rendered[0], sr

    # Equal-power average
    maxlen = max(len(a) for a in rendered)
    out = np.zeros(maxlen, dtype=np.float32)

    for a in rendered:
        if len(a) < maxlen:
            a = np.pad(a, (0, maxlen - len(a)))
        out += a

    out /= float(len(rendered))
    out = np.clip(out, -1.0, 1.0)

    return out, sr


def _encode_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _encode_flac_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="FLAC")
    return buf.getvalue()


def _encode_ogg_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    try:
        sf.write(buf, audio, sr, format="OGG", subtype="VORBIS")
        return buf.getvalue()
    except Exception:
        seg = AudioSegment(
            (np.clip(audio, -1, 1) * 32767.0).astype(np.int16).tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1,
        )
        out = io.BytesIO()
        seg.export(out, format="ogg", bitrate="192k")
        return out.getvalue()


def _encode_mp3_bytes(audio: np.ndarray, sr: int) -> bytes:
    """
    Stable MP3 encoder for TTS.
    Prevents quiet phoneme trimming (like 'kokoro').
    """
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)

    seg = AudioSegment(
        pcm16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1,
    )

    seg = seg.apply_gain(+3)

    out = io.BytesIO()

    seg.export(
        out,
        format="mp3",
        bitrate="192k",
        parameters=["-af", "aresample=async=1"]
    )

    return out.getvalue()


def encode_audio(audio: np.ndarray, sr: int, fmt: str) -> Tuple[bytes, str]:
    fmt = (fmt or "wav").lower()

    if fmt == "wav":
        return _encode_wav_bytes(audio, sr), "audio/wav"

    if fmt == "flac":
        return _encode_flac_bytes(audio, sr), "audio/flac"

    if fmt == "ogg":
        return _encode_ogg_bytes(audio, sr), "audio/ogg"

    if fmt == "mp3":
        return _encode_mp3_bytes(audio, sr), "audio/mpeg"

    return _encode_wav_bytes(audio, sr), "audio/wav"

def maybe_save(audio: np.ndarray, sr: int, basename: str, enable: bool) -> Optional[str]:
    if not enable:
        return None

    os.makedirs(settings.save_dir, exist_ok=True)
    path = os.path.join(settings.save_dir, f"{basename}.wav")

    sf.write(path, audio, sr, format="WAV", subtype="PCM_16")
    return path
