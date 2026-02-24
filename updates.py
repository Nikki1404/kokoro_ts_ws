for fastapi openai used this code 
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
#main.py
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

and for websocket used this code 
#config.yaml-
format: "f32"
speed: 1.0

voice: "af_heart"
lang_code: "a"

output_dir: "out_audio"
sample_rate: 24000

server:
  host: "0.0.0.0"
  port: 4000
  reload: false

logging:
  level: "INFO"
  save_logs: false
  log_dir: "logs"

pipeline:
  split_pattern: "\\n+"
  save_wav: false

#ws_kokoro_server.py-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import os
import traceback
import logging
from typing import List
from io import BytesIO
from datetime import datetime
import numpy as np
import yaml
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
import uvicorn
import soundfile as sf
from pydub import AudioSegment

try:
    import torch
except Exception:
    torch = None

from kokoro import KPipeline

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

logging.basicConfig(level=CONFIG["logging"]["level"])

SR = CONFIG["sample_rate"]
app = FastAPI()
pipeline = None

# Thread pool for encoding
executor = ThreadPoolExecutor(max_workers=4)

# Chunking config
CHUNK_ENABLED = CONFIG.get("chunking", {}).get("enabled", True)
WORD_THRESHOLD = int(CONFIG.get("chunking", {}).get("word_threshold", 20))


def as_numpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32).reshape(-1)
    return np.asarray(x, dtype=np.float32).reshape(-1)

def get_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def linear_to_mulaw(x, mu=255):
    """Convert float PCM (-1..1) to µ-law 8-bit."""
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return ((y + 1) * 127.5).astype(np.uint8)


def resample_to_8k(audio_float, orig_sr=24000):
    """Downsample float32 PCM to 8000 Hz."""
    import librosa
    return librosa.resample(audio_float, orig_sr=orig_sr, target_sr=8000)


def chunk_text(text: str, max_words: int = 20) -> list[str]:
    text = text.strip()
    if not text:
        return []

    total_words = len(text.split())
    if total_words <= max_words:
        return [text]

    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        words = sentence.split()
        if len(words) <= max_words:
            chunks.append(sentence)
            continue

        for i in range(0, len(words), max_words):
            chunks.append(" ".join(words[i:i + max_words]))

    return chunks


@app.on_event("startup")
async def _init():
    global pipeline
    device = get_device()
    pipeline = KPipeline(
        lang_code=CONFIG["lang_code"],
        device=device
    )
    logging.info(f" Kokoro TTS initialized (device={device}, lang={CONFIG['lang_code']})")

@app.get("/", response_class=PlainTextResponse)
def root():
    return (
        "Kokoro TTS WebSocket server\n\n"
        "Formats: f32 | s16 | wav | mp3 | ogg | flac | mulaw | pcm8 | s16_8k\n"
    )


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    logging.info("connection opened")

    try:
        while True:
            start = time.time()
            print(f"----------START TIME: {start}------------")

            req = await websocket.receive_text()
            cfg = json.loads(req)

            text = str(cfg.get("text", "")).strip()
            voice = str(cfg.get("voice", CONFIG["voice"]))
            speed = float(cfg.get("speed", CONFIG["speed"]))
            fmt = str(cfg.get("format", CONFIG["format"])).lower()

            if not text:
                await websocket.send_text(json.dumps({"type": "done", "error": "empty text"}))
                continue

            await websocket.send_text(json.dumps({
                "type": "meta",
                "sample_rate": SR,
                "channels": 1,
                "sample_format": fmt
            }))

            # Chunking
            chunks = chunk_text(text, WORD_THRESHOLD) if CHUNK_ENABLED else [text]

            only_pipeline_start = time.time()
            print(f"----------ONLY PIPELINE START TIME: {only_pipeline_start}------------")

            t0 = time.perf_counter()
            ttfa_sent = False
            segments = 0
            audio_total_s = 0.0
            buf: List[np.ndarray] = []

            # Process each chunk
            for chunk in chunks:
                gen = pipeline(chunk, voice=voice, speed=speed,
                               split_pattern=CONFIG["pipeline"]["split_pattern"])

                for (_gs, _ps, audio) in gen:
                    now = time.perf_counter()

                    if not ttfa_sent:
                        await websocket.send_text(json.dumps(
                            {"type": "ttfa", "ms": (now - t0) * 1000.0}
                        ))
                        ttfa_sent = True

                    a = as_numpy(audio)
                    buf.append(a)

                    segments += 1
                    audio_total_s += a.size / SR

                    # Minimal processing + immediate streaming
                    if fmt == "f32":
                        await websocket.send_bytes(a.tobytes())

                    elif fmt == "s16":
                        pcm16 = (np.clip(a, -1, 1) * 32767.0).astype(np.int16)
                        await websocket.send_bytes(pcm16.tobytes())

                    elif fmt in {"mulaw", "pcm8", "s16_8k"}:
                        # Resample to 8k
                        a_8k = resample_to_8k(a, orig_sr=SR)

                        if fmt == "mulaw":
                            encoded = linear_to_mulaw(a_8k)
                            await websocket.send_bytes(encoded.tobytes())

                        elif fmt == "pcm8":
                            await websocket.send_bytes(a_8k.astype(np.float32).tobytes())

                        elif fmt == "s16_8k":
                            pcm16_8k = (np.clip(a_8k, -1, 1) * 32767.0).astype(np.int16)
                            await websocket.send_bytes(pcm16_8k.tobytes())

            print(f"---------------ONLY PIPELINE Time taken: {time.time() - only_pipeline_start} seconds-----------")

            total_ms = (time.perf_counter() - t0) * 1000.0
            rtf = (total_ms / 1000.0) / max(1e-6, audio_total_s)

            # Threaded encoding (mp3, ogg, flac)
            if fmt in {"wav", "mp3", "ogg", "flac"}:
                full_audio = np.concatenate(buf) if len(buf) > 1 else buf[0]

                # Direct WAV creation
                wav_io = BytesIO()
                sf.write(wav_io, np.clip(full_audio, -1, 1), SR,
                         format="WAV", subtype="PCM_16")
                wav_bytes = wav_io.getvalue()

                if fmt == "wav":
                    await websocket.send_bytes(wav_bytes)

                else:
                    loop = asyncio.get_running_loop()

                    def encode_fn():
                        audio_seg = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")
                        out = BytesIO()
                        audio_seg.export(out, format=fmt)
                        return out.getvalue()

                    encoded_bytes = await loop.run_in_executor(executor, encode_fn)
                    await websocket.send_bytes(encoded_bytes)

            # DONE message
            await websocket.send_text(json.dumps({
                "type": "done",
                "total_ms": total_ms,
                "audio_ms": audio_total_s * 1000.0,
                "segments": segments,
                "rtf": rtf,
                "error": None
            }))

            print(f"---------------Time taken: {time.time() - start} seconds-----------")

    except WebSocketDisconnect:
        logging.warning("Client disconnected.")
    except Exception as e:
        logging.error(traceback.format_exc())
        try:
            await websocket.send_text(json.dumps(
                {"type": "done", "error": str(e)}
            ))
        except:
            pass
        await websocket.close()
    finally:
        logging.info("connection closed")

if __name__ == "__main__":
    uvicorn.run(
        "ws_kokoro_server:app",
        host=CONFIG["server"]["host"],
        port=int(CONFIG["server"]["port"]),
        reload=CONFIG["server"]["reload"]
    )

so tell my why when testin websocket via this script 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio, json
import numpy as np
import websockets
import sounddevice as sd

async def tts_once(url, text, voice="af_heart", speed=1.0, fmt="f32"):
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"text": text, "voice": voice, "speed": speed, "format": fmt}))
        sr = 24000
        stream = None
        dtype = np.float32 if fmt == "f32" else np.int16
        try:
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    a = np.frombuffer(msg, dtype=dtype)
                    if fmt == "s16":  # convert to float for playback
                        a = (a.astype(np.float32) / 32767.0)
                    if stream is None:
                        stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
                        stream.start()
                    stream.write(a.reshape(-1, 1))
                else:
                    data = json.loads(msg)
                    t = data.get("type")
                    if t == "meta":
                        sr = int(data["sample_rate"])
                        print(f"[meta] sr={sr}, fmt={data['sample_format']}")
                    elif t == "ttfa":
                        print(f"[ttfa] {data['ms']:.1f} ms")
                    elif t == "done":
                        if data.get("error"): print("[done:ERROR]", data["error"])
                        else:
                            print(f"[done] gen={data['total_ms']:.1f} ms, audio={data['audio_ms']:.1f} ms, "
                                  f"segments={data['segments']}, rtf={data['rtf']:.3f}")
                        break
        finally:
            if stream is not None:
                stream.stop(); stream.close()

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="wss://kokoro-ws-150916788856.us-central1.run.app/ws")
    ap.add_argument("--text", default=None, help="If provided, runs once with this text; otherwise interactive.")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--fmt", default="f32", choices=["f32","s16"])
    args = ap.parse_args()

    async def run():
        # Single-shot mode only if --text is provided
        if args.text is not None:
            await tts_once(args.url, args.text, args.voice, args.speed, args.fmt)
            return

        # Default: interactive mode (no flags needed)
        print("Kokoro WS client (interactive). Type text and press Enter. /q to quit.")
        while True:
            try:
                line = input("text> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line in {"/q", "/quit", "/exit"}:
                break
            try:
                await tts_once(args.url, line, args.voice, args.speed, args.fmt)
            except KeyboardInterrupt:
                print("\n[info] cancelled current utterance")
            except Exception as e:
                print(f"[warn] send/play failed: {e}")

    asyncio.run(run())

convert kokoro word to tts but 
when testing for openai via this
from openai import OpenAI

client = OpenAI(
    base_url="https://kokoro-openai-tts-150916788856.us-central1.run.app/v1",
    api_key="not-needed",
)

with client.audio.speech.with_streaming_response.create(
    model="kokoro",
    voice="af_sky",
    input="this is kokoro",
    response_format="mp3",
) as resp:
    resp.stream_to_file("output5.mp3")

print("Saved -> output5.mp3")
kokoro word is not getting converted to tts
