import io
import logging
import os
import threading
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from kokoro import Kokoro

logger = logging.getLogger("kokoro-api")


KOKORO_LANG = os.getenv("KOKORO_LANG", "a")
KOKORO_DEFAULT_VOICE = os.getenv("KOKORO_DEFAULT_VOICE", "af_heart")
KOKORO_PRELOAD_VOICES = os.getenv("KOKORO_PRELOAD_VOICES", "").strip()
KOKORO_WARMUP_TEXT = os.getenv("KOKORO_WARMUP_TEXT", "warmup")

KOKORO_DEVICE = os.getenv("KOKORO_DEVICE", "").strip().lower()

# We will mount it at /v1 
router = APIRouter()

# -----------------------------
# Model singleton with lock
# -----------------------------
_MODEL = None
_MODEL_LOCK = threading.Lock()


def _resolve_device() -> str:
    if KOKORO_DEVICE in ("cuda", "cpu"):
        return KOKORO_DEVICE
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model() -> Kokoro:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        device = _resolve_device()
        logger.info("Loading Kokoro TTS model | device=%s | lang=%s", device, KOKORO_LANG)

        # Kokoro constructor varies slightly by version; this is the common pattern for kokoro-tts
        _MODEL = Kokoro(device=device, lang=KOKORO_LANG)

        # Optional warmup / preload voices to reduce first-request latency
        if KOKORO_PRELOAD_VOICES:
            voices = [v.strip() for v in KOKORO_PRELOAD_VOICES.split() if v.strip()]
            if voices:
                logger.info("Preloading voices: %s", ", ".join(voices))
                for v in voices:
                    try:
                        _MODEL.create(text=KOKORO_WARMUP_TEXT, voice=v)
                    except Exception as e:
                        logger.warning("Voice preload failed for '%s': %s", v, str(e))

        logger.info("✅ Kokoro loaded")
        return _MODEL

class AudioSpeechIn(BaseModel):
    # OpenAI SDK sends these fields (model/voice/input) for audio.speech
    model: Optional[str] = Field(default="kokoro")
    voice: Optional[str] = Field(default=None, description="Kokoro voice id (e.g., af_heart)")
    input: str = Field(..., description="Text to synthesize")

    # Optional extras
    instructions: Optional[str] = Field(default=None, description="(Optional) ignored or used to select voice/style")
    format: Optional[str] = Field(default="wav", description="Return format. This server returns WAV only.")
    speed: Optional[float] = Field(default=None, description="(Optional) not used unless your Kokoro version supports it")


@router.get("/health")
def health():
    return {
        "status": "ok",
        "gpu_available": bool(torch.cuda.is_available()),
        "default_voice": KOKORO_DEFAULT_VOICE,
        "lang": KOKORO_LANG,
        "preload_voices": KOKORO_PRELOAD_VOICES,
    }


@router.post("/audio/speech")
async def audio_speech(body: AudioSpeechIn):
    """
    OpenAI-compatible endpoint:
      POST /v1/audio/speech
      POST /v1/tts/audio/speech  (also supported via mounting)
    Returns: audio/wav stream
    """
    try:
        text = (body.input or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="input text is empty")

        voice = (body.voice or "").strip() or KOKORO_DEFAULT_VOICE

        model = get_model()
        wav, sr = model.create(text=text, voice=voice)

        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().numpy()

        wav = np.asarray(wav).flatten()

        # Write WAV to in-memory buffer
        buffer = io.BytesIO()
        sf.write(buffer, wav, sr, format="WAV")
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("TTS failed")
        raise HTTPException(status_code=500, detail=str(e))
