from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import torch
import numpy as np

from kokoro import KPipeline

router = APIRouter(prefix="/v1")

MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        MODEL = KPipeline(lang_code='a', device=DEVICE)
    return MODEL


def audio_generator(text: str, voice: str = 'af_heart'):
    model = get_model()
    generator = model(text, voice=voice, speed=1, split_pattern=r'\n+')

    for _, _, audio in generator:
        # Convert float32 (-1..1) → int16 PCM
        audio_np = np.asarray(audio, dtype=np.float32)
        pcm16 = (audio_np * 32767.0).astype(np.int16)
        yield pcm16.tobytes()


class TextInput(BaseModel):
    input: str = Field(..., description="Text to synthesize")


@router.post("/tts/audio/speech")
async def audio_speech(body: TextInput):
    try:
        text = (body.input or "").strip()
        return StreamingResponse(
            audio_generator(text),
            media_type="audio/L16"  # Raw PCM 16-bit
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#client.py-
from openai import OpenAI

client = OpenAI(
    base_url="https://tts-kokoro-openapi-150916788856.europe-west1.run.app/v1/tts",
    api_key="not-needed",
)

with client.audio.speech.with_streaming_response.create(
    model="",
    voice="",
    input="Hello world from kokoro openapi deployed on gcp without docker testing",
) as resp:

    with open("output.pcm", "wb") as f:
        for chunk in resp.iter_bytes():
            f.write(chunk)

print("Saved -> output.pcm")
