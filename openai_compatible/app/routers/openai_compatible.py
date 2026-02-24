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

from kokoro import KPipeline   # ✅ Using GitHub Kokoro

router = APIRouter(prefix="/v1")

MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        logging.info(f"Loading Kokoro KPipeline on {DEVICE}...")

        MODEL = KPipeline(
            lang_code=os.getenv("KOKORO_LANG", "a"),
            device=DEVICE
        )

        logging.info("✅ Kokoro KPipeline loaded")

    return MODEL


class AudioSpeechIn(BaseModel):
    model: Optional[str] = Field(default="kokoro")
    voice: Optional[str] = Field(default="af_heart")
    input: str
    instructions: Optional[str] = None


@router.post("/audio/speech")
async def audio_speech(body: AudioSpeechIn):
    try:
        text = (body.input or "").strip()
        voice = body.voice or os.getenv("KOKORO_DEFAULT_VOICE", "af_heart")

        pipeline = get_model()

        # Generate audio via streaming generator
        generator = pipeline(
            text,
            voice=voice
        )

        audio_chunks = []

        for chunk in generator:
            audio = chunk.audio
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            audio_chunks.append(audio)

        if not audio_chunks:
            raise HTTPException(status_code=500, detail="No audio generated")

        wav = np.concatenate(audio_chunks, axis=-1).flatten()
        sr = 24000  # Kokoro default sample rate

        buffer = io.BytesIO()
        sf.write(buffer, wav, sr, format="WAV")
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"},
        )

    except Exception as e:
        logging.exception("TTS generation failed")
        raise HTTPException(status_code=500, detail=str(e))
