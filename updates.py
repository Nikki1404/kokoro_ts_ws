# app/core/mappings.py
OPENAI_MODEL_MAP = {
    "tts-1": "kokoro-v1_0",
    "tts-1-hd": "kokoro-v1_0",
    "kokoro": "kokoro-v1_0",
}

OPENAI_VOICE_MAP = {
    "alloy": "am_v0adam",
    "ash": "af_v0nicole",
    "coral": "bf_v0emma",
    "echo": "af_v0bella",
    "fable": "af_sarah",
    "onyx": "bm_george",
    "nova": "bf_isabella",
    "sage": "am_michael",
    "shimmer": "af_sky",
}

# app/core/config.py  (Pydantic v2)
from functools import lru_cache
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # tell pydantic-settings to load .env
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Server
    app_name: str = "Kokoro OpenAI-Compatible TTS"
    debug: bool = Field(False, alias="DEBUG")
    host: str = Field("0.0.0.0", alias="HOST")
    port: int = Field(8081, alias="PORT")

    # CORS
    cors_enabled: bool = Field(True, alias="CORS_ENABLED")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], alias="CORS_ORIGINS")

    # Kokoro defaults
    lang_code: str = Field("a", alias="KOKORO_LANG_CODE")              # a(US-EN), b(UK-EN), h(Hindi), …
    default_voice: str = Field("af_heart", alias="KOKORO_DEFAULT_VOICE")
    default_speed: float = Field(1.0, alias="KOKORO_DEFAULT_SPEED")
    default_sample_rate: int = Field(24000, alias="KOKORO_SAMPLE_RATE")

    # Storage
    save_audio: bool = Field(True, alias="SAVE_AUDIO")
    save_dir: str = Field("app/assets/out", alias="SAVE_DIR")

    # Formats
    allowed_formats: List[str] = Field(
        default_factory=lambda: ["wav", "mp3", "ogg", "flac"],
        alias="ALLOWED_FORMATS",

    )

@lru_cache
def get_settings() -> "Settings":
    return Settings()

settings = get_settings()

# app/routers/openai_compatible.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional

from app.core.config import settings
from app.tts.kokoro_engine import synthesize_np, encode_audio, maybe_save

router = APIRouter(prefix="/v1")

class AudioSpeechIn(BaseModel):
    model: str = Field("tts-1", description="Ignored; kept for compatibility")
    voice: Optional[str] = Field(None, description="Kokoro voice id (e.g., af_heart or af_sky+af_bella)")
    input: str = Field(..., description="Text to synthesize")
    response_format: Optional[str] = Field("wav", description="wav|mp3|ogg|flac")
    speed: Optional[float] = Field(None, description="1.0 = normal")
    stream: Optional[bool] = Field(False, description="If true, returns chunked bytes")
    lang_code: Optional[str] = Field(None, description="Kokoro language code (default from server)")
    sample_rate: Optional[int] = Field(None, description="Default from server")
    save: Optional[bool] = Field(None, description="Override server save_audio")

@router.post("/audio/speech")
async def audio_speech(body: AudioSpeechIn):
    fmt = (body.response_format or "wav").lower()
    if fmt not in settings.allowed_formats and fmt != "wav":
        raise HTTPException(status_code=400, detail=f"Unsupported response_format='{fmt}'")

    text = (body.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty 'input'")

    try:
        audio, sr = synthesize_np(
            text=text,
            voice=body.voice,
            speed=body.speed if body.speed is not None else settings.default_speed,
            lang_code=body.lang_code or settings.lang_code,
            sample_rate=body.sample_rate or settings.default_sample_rate,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kokoro synth failed: {e}")

    # Optional save (never break request if it fails)
    try:
        _ = maybe_save(
            audio=audio,
            sr=sr,
            basename="out",
            enable=body.save if body.save is not None else settings.save_audio,
        )
    except Exception:
        pass

    try:
        blob, ctype = encode_audio(audio, sr, fmt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {e}")

    if body.stream:
        def _iter(b: bytes, sz: int = 64 * 1024):
            for i in range(0, len(b), sz):
                yield b[i:i+sz]
        return StreamingResponse(_iter(blob), media_type=ctype)

    return Response(content=blob, media_type=ctype)


# app/tts/kokoro_engine.py
import io
import os
from typing import Optional, Tuple

import numpy as np

from app.core.config import settings
from kokoro import KPipeline

import soundfile as sf
from pydub import AudioSegment

_PIPELINE: Optional[KPipeline] = None
_LANG_IN_USE: Optional[str] = None

def _get_pipeline(lang_code: str) -> KPipeline:
    global _PIPELINE, _LANG_IN_USE
    if _PIPELINE is None or _LANG_IN_USE != lang_code:
        _PIPELINE = KPipeline(lang_code=lang_code)
        _LANG_IN_USE = lang_code
    return _PIPELINE

def _as_float32_mono(x) -> np.ndarray:
    import numpy as _np
    try:
        import torch as _torch
        if isinstance(x, _torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    a = _np.asarray(x, dtype=_np.float32).reshape(-1)
    return a

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

    voices = [v.strip() for v in (voice or "").split("+") if v.strip()] or [settings.default_voice]
    rendered = []

    for v in voices:
        chunks = []
        gen = pipe(text, voice=v, speed=float(speed), split_pattern=r"\n+")
        for (_gs, _ps, audio) in gen:
            chunks.append(_as_float32_mono(audio))
        if chunks:
            rendered.append(np.concatenate(chunks) if len(chunks) > 1 else chunks[0])

    if not rendered:
        return np.zeros(0, dtype=np.float32), sr
    if len(rendered) == 1:
        return rendered[0], sr

    # Simple equal-power average (pad to max length first)
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
            (audio * 32767.0).astype(np.int16).tobytes(),
            frame_rate=sr, sample_width=2, channels=1,
        )
        out = io.BytesIO()
        seg.export(out, format="ogg")
        return out.getvalue()

def _encode_mp3_bytes(audio: np.ndarray, sr: int) -> bytes:
    seg = AudioSegment(
        (audio * 32767.0).astype(np.int16).tobytes(),
        frame_rate=sr, sample_width=2, channels=1,
    )
    out = io.BytesIO()
    seg.export(out, format="mp3")
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
    # Fallback
    return _encode_wav_bytes(audio, sr), "audio/wav"

def maybe_save(audio: np.ndarray, sr: int, basename: str, enable: bool) -> Optional[str]:
    if not enable:
        return None
    os.makedirs(settings.save_dir, exist_ok=True)
    path = os.path.join(settings.save_dir, f"{basename}.wav")
    sf.write(path, audio, sr, format="WAV", subtype="PCM_16")
    return path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.routers.openai_compatible import router as openai_router
from kokoro import KPipeline
import os

app = FastAPI(title=settings.app_name, debug=settings.debug)

@app.get("/debug/gpu")
def gpu_debug():
    try:
        import onnxruntime as ort
        return {"providers": ort.get_available_providers()}
    except Exception as e:
        return {"error": str(e)}
        

if settings.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(openai_router)


@app.on_event("startup")
async def preload_kokoro_voices():
    """Automatically preload Kokoro voices on startup (first run)."""
    voices = os.getenv("KOKORO_PRELOAD_VOICES", "").split()
    if not voices:
        print("No KOKORO_PRELOAD_VOICES defined, skipping preload.")
        return
    pipe = KPipeline(lang_code=settings.lang_code)
    print(f"Preloading Kokoro voices: {voices}")
    for v in voices:
        try:
            list(pipe("hello world", voice=v))
            print(f" Cached voice: {v}")
        except Exception as e:
            print(f"Failed to preload {v}: {e}")


@app.get("/healthz")
def healthz():
    return {"ok": True, "lang": settings.lang_code, "voice": settings.default_voice}
