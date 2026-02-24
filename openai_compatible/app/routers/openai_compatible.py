# app/routers/openai_compatible.py

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import logging
import torch
import soundfile as sf
import io
import numpy as np
import os

from kokoro import Kokoro

router = APIRouter(prefix="/v1")

MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        logging.info(f"Loading Kokoro model on {DEVICE}...")

        MODEL = Kokoro(
            device=DEVICE,
            lang=os.getenv("KOKORO_LANG", "a")
        )

        logging.info("✅ Kokoro model loaded")

    return MODEL


class AudioSpeechIn(BaseModel):
    model: Optional[str] = Field(default="kokoro")
    voice: Optional[str] = Field(default="af_heart")
    input: str
    instructions: Optional[str] = None


# ✅ IMPORTANT: no "/tts" here
@router.post("/audio/speech")
async def audio_speech(body: AudioSpeechIn):
    try:
        text = (body.input or "").strip()
        voice = body.voice or os.getenv("KOKORO_DEFAULT_VOICE", "af_heart")

        model = get_model()

        wav, sr = model.create(
            text=text,
            voice=voice
        )

        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()

        if isinstance(wav, list):
            wav = np.concatenate(wav, axis=-1)

        wav = wav.flatten()

        buffer = io.BytesIO()
        sf.write(buffer, wav, sr, format="WAV")
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
