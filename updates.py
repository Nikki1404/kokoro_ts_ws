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

# app/routers/openai_compatible.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import io
import numpy as np
import soundfile as sf
from pydub import AudioSegment

from app.core.config import settings
from app.tts.kokoro_engine import _get_pipeline, _as_float32_mono

router = APIRouter(prefix="/v1")


class AudioSpeechIn(BaseModel):
    model: str = Field("tts-1")
    voice: Optional[str] = None
    input: str
    response_format: Optional[str] = "wav"
    speed: Optional[float] = None
    stream: Optional[bool] = False
    lang_code: Optional[str] = None
    sample_rate: Optional[int] = None
    save: Optional[bool] = None


@router.post("/audio/speech")
async def audio_speech(body: AudioSpeechIn):

    fmt = (body.response_format or "wav").lower()
    text = (body.input or "").strip()

    if not text:
        raise HTTPException(status_code=400, detail="Empty 'input'")

    voice = body.voice or settings.default_voice
    speed = body.speed if body.speed is not None else settings.default_speed
    lang_code = body.lang_code or settings.lang_code
    sr = body.sample_rate or settings.default_sample_rate

    pipe = _get_pipeline(lang_code)

    # 🔥 IMPORTANT: MATCH WEBSOCKET BEHAVIOR
    from app.tts.kokoro_engine import _chunk_text

    chunks = _chunk_text(text, max_words=20)

    def generate():

        for chunk in chunks:

            gen = pipe(
                chunk,
                voice=voice,
                speed=float(speed),
                split_pattern=r"\n+",
            )

            for (_gs, _ps, audio) in gen:

                arr = _as_float32_mono(audio)

                if arr.size == 0:
                    continue

                if fmt == "wav":
                    buf = io.BytesIO()
                    sf.write(buf, arr, sr, format="WAV", subtype="PCM_16")
                    yield buf.getvalue()

                elif fmt == "mp3":
                    pcm16 = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
                    seg = AudioSegment(
                        pcm16.tobytes(),
                        frame_rate=sr,
                        sample_width=2,
                        channels=1,
                    )
                    out = io.BytesIO()
                    seg.export(out, format="mp3", bitrate="192k")
                    yield out.getvalue()
    return StreamingResponse(
        generate(),
        media_type=media_type_map.get(fmt, "audio/wav"),
    )
# kokoro_engine.py

# main.py
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
        import torch
        return {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
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


#kokoro_engine.py-
import io
import os
import re
from typing import Optional, Tuple, List

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from app.core.config import settings

# Kokoro import
from kokoro import KPipeline

_PIPELINE: Optional[KPipeline] = None
_LANG_IN_USE: Optional[str] = None
_DEVICE_IN_USE: Optional[str] = None


# -----------------------------
# DEVICE DETECTION (TORCH BASED)
# -----------------------------
def _get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# -----------------------------
# PIPELINE LOADER (DEVICE AWARE)
# -----------------------------
def _get_pipeline(lang_code: str) -> KPipeline:
    global _PIPELINE, _LANG_IN_USE, _DEVICE_IN_USE

    device = _get_device()

    if _PIPELINE is None or _LANG_IN_USE != lang_code or _DEVICE_IN_USE != device:
        print(f"[Kokoro] Initializing pipeline (lang={lang_code}, device={device})")
        _PIPELINE = KPipeline(lang_code=lang_code, device=device)
        _LANG_IN_USE = lang_code
        _DEVICE_IN_USE = device

    return _PIPELINE


# -----------------------------
# TEXT CHUNKING (sentence-aware)
# matches websocket behavior
# -----------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _chunk_text(text: str, max_words: int = 20) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    words = text.split()
    if len(words) <= max_words:
        return [text]

    sentences = _SENT_SPLIT.split(text)
    chunks: List[str] = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        w = s.split()
        if len(w) <= max_words:
            chunks.append(s)
        else:
            for i in range(0, len(w), max_words):
                chunks.append(" ".join(w[i:i + max_words]))

    return chunks if chunks else [text]


# -----------------------------
# AUDIO NORMALIZATION
# -----------------------------
def _as_float32_mono(x) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x, dtype=np.float32).reshape(-1)


# -----------------------------
# SYNTHESIS
# -----------------------------
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

    # multi-voice: "af_sky+af_bella"
    voices = [v.strip() for v in (voice or "").split("+") if v.strip()] or [settings.default_voice]

    # chunk text
    text_chunks = _chunk_text(text, max_words=20)

    rendered: List[np.ndarray] = []

    for v in voices:
        buf: List[np.ndarray] = []

        for chunk in text_chunks:
            gen = pipe(chunk, voice=v, speed=float(speed), split_pattern=r"\n+")
            for (_gs, _ps, audio) in gen:
                buf.append(_as_float32_mono(audio))

        if buf:
            rendered.append(np.concatenate(buf) if len(buf) > 1 else buf[0])

    if not rendered:
        return np.zeros(0, dtype=np.float32), sr

    if len(rendered) == 1:
        return rendered[0], sr

    # mix multi voice
    maxlen = max(len(a) for a in rendered)
    out = np.zeros(maxlen, dtype=np.float32)
    for a in rendered:
        if len(a) < maxlen:
            a = np.pad(a, (0, maxlen - len(a)))
        out += a
    out /= float(len(rendered))
    out = np.clip(out, -1.0, 1.0)

    return out, sr


# -----------------------------
# ENCODERS
# -----------------------------
def _encode_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1, 1), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _encode_flac_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1, 1), sr, format="FLAC")
    return buf.getvalue()


def _encode_ogg_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    try:
        sf.write(buf, np.clip(audio, -1, 1), sr, format="OGG", subtype="VORBIS")
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
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    seg = AudioSegment(
        pcm16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1,
    )

    # mild boost to preserve soft consonants
    seg = seg.apply_gain(+3)

    out = io.BytesIO()
    seg.export(out, format="mp3", bitrate="192k", parameters=["-af", "aresample=async=1"])
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
    sf.write(path, np.clip(audio, -1, 1), sr, format="WAV", subtype="PCM_16")
    return path


#Dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ARG http_proxy
ARG https_proxy

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV KOKORO_LANG=a
ENV KOKORO_DEFAULT_VOICE=af_heart
ENV KOKORO_PRELOAD_VOICES="af_heart af_bella af_sky"

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /app/
RUN python3 -m pip install --no-cache-dir -r requirements.txt

RUN python3 -m pip install --no-cache-dir \
    git+https://github.com/nvidia/kokoro.git

COPY . /app/

EXPOSE 8081
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8081"]


from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import io
import numpy as np
import soundfile as sf
from pydub import AudioSegment

from app.core.config import settings
from app.tts.kokoro_engine import _get_pipeline, _as_float32_mono, _chunk_text

router = APIRouter(prefix="/v1")


class AudioSpeechIn(BaseModel):
    model: str = Field("tts-1", description="Ignored; kept for compatibility")
    voice: Optional[str] = Field(None, description="Kokoro voice id (e.g., af_heart or af_sky+af_bella)")
    input: str = Field(..., description="Text to synthesize")
    response_format: Optional[str] = Field("wav", description="wav|mp3|ogg|flac")
    speed: Optional[float] = Field(None, description="1.0 = normal")
    stream: Optional[bool] = Field(False, description="Kept for compatibility; StreamingResponse is used anyway")
    lang_code: Optional[str] = Field(None, description="Kokoro language code (default from server)")
    sample_rate: Optional[int] = Field(None, description="Default from server")
    save: Optional[bool] = Field(None, description="Unused in streaming version")


@router.post("/audio/speech")
async def audio_speech(body: AudioSpeechIn):
    fmt = (body.response_format or "wav").lower()
    if fmt not in settings.allowed_formats and fmt != "wav":
        raise HTTPException(status_code=400, detail=f"Unsupported response_format='{fmt}'")

    text = (body.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty 'input'")

    voice = body.voice or settings.default_voice
    speed = body.speed if body.speed is not None else settings.default_speed
    lang_code = body.lang_code or settings.lang_code
    sr = int(body.sample_rate or settings.default_sample_rate)

    pipe = _get_pipeline(lang_code)

    # Match websocket-style pre-chunking to avoid word drops
    chunks = _chunk_text(text, max_words=20)

    media_type_map = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
    }

    def generate():
        for chunk in chunks:
            gen = pipe(chunk, voice=voice, speed=float(speed), split_pattern=r"\n+")
            for (_gs, _ps, audio) in gen:
                arr = _as_float32_mono(audio)
                if arr.size == 0:
                    continue

                if fmt == "wav":
                    buf = io.BytesIO()
                    sf.write(buf, arr, sr, format="WAV", subtype="PCM_16")
                    yield buf.getvalue()

                elif fmt == "flac":
                    buf = io.BytesIO()
                    sf.write(buf, arr, sr, format="FLAC")
                    yield buf.getvalue()

                elif fmt == "ogg":
                    # prefer soundfile; if it fails, pydub fallback
                    try:
                        buf = io.BytesIO()
                        sf.write(buf, arr, sr, format="OGG", subtype="VORBIS")
                        yield buf.getvalue()
                    except Exception:
                        pcm16 = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
                        seg = AudioSegment(
                            pcm16.tobytes(),
                            frame_rate=sr,
                            sample_width=2,
                            channels=1,
                        )
                        out = io.BytesIO()
                        seg.export(out, format="ogg", bitrate="192k")
                        yield out.getvalue()

                elif fmt == "mp3":
                    pcm16 = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
                    seg = AudioSegment(
                        pcm16.tobytes(),
                        frame_rate=sr,
                        sample_width=2,
                        channels=1,
                    )
                    out = io.BytesIO()
                    seg.export(out, format="mp3", bitrate="192k")
                    yield out.getvalue()

                else:
                    # fallback wav
                    buf = io.BytesIO()
                    sf.write(buf, arr, sr, format="WAV", subtype="PCM_16")
                    yield buf.getvalue()

    return StreamingResponse(
        generate(),
        media_type=media_type_map.get(fmt, "audio/wav"),
    )


(kokoro_env) PS C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\fastapi_impl\app> python .\test_openai.py
Traceback (most recent call last):
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\fastapi_impl\app\test_openai.py", line 8, in <module>
    response = client.audio.speech.create(
        model="kokoro",
    ...<2 lines>...
        response_format="mp3",
    )
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\openai\resources\audio\speech.py", line 104, in create
    return self._post(
           ~~~~~~~~~~^
        "/audio/speech",
        ^^^^^^^^^^^^^^^^
    ...<15 lines>...
        cast_to=_legacy_response.HttpxBinaryResponseContent,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\openai\_base_client.py", line 1297, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
                           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\openai\_base_client.py", line 1070, in request
    raise self._make_status_error_from_response(err.response) from None
openai.InternalServerError: Internal Server Error
(kokoro_env) PS C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\fastapi_impl\app> python .\test_openai.py
Traceback (most recent call last):
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\fastapi_impl\app\test_openai.py", line 8, in <module>
    response = client.audio.speech.create(
        model="kokoro",
    ...<2 lines>...
        response_format="mp3",
    )
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\openai\resources\audio\speech.py", line 104, in create
    return self._post(
           ~~~~~~~~~~^
        "/audio/speech",
        ^^^^^^^^^^^^^^^^
    ...<15 lines>...
        cast_to=_legacy_response.HttpxBinaryResponseContent,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\openai\_base_client.py", line 1297, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
                           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\openai\_base_client.py", line 1070, in request
    raise self._make_status_error_from_response(err.response) from None
openai.InternalServerError: Internal Server Error
(kokoro_env) PS C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\fastapi_impl\app> python .\test_openai.py
Saved -> output_fixed1.mp3
(kokoro_env) PS C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\fastapi_impl\app> python .\test_openai.py
Traceback (most recent call last):
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\fastapi_impl\app\test_openai.py", line 16, in <module>
    f.write(response.content)
            ^^^^^^^^
NameError: name 'response' is not defined


when running this 
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8081/v1",
    api_key="not-needed",
)

with client.audio.speech.with_streaming_response.create(
    model="kokoro",
    voice="af_heart",
    input="Hello from Kokoro using KPipeline",
) as resp:
    resp.stream_to_file("output_fixed1.wav")

with open("output_fixed1.wav", "wb") as f:
    f.write(response.content)

print("Saved -> output_fixed1.wav")
