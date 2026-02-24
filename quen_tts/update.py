# app/routers/openai_compatible.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import logging
from qwen_tts import Qwen3TTSModel
import torch
import soundfile as sf
import io
import numpy as np

router  = APIRouter(prefix="/v1")


MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        if torch.cuda.is_available():
            DEVICE = "cuda"
            dtype = torch.bfloat16
        else:
            DEVICE = "cpu"
            dtype = torch.bfloat32

        logging.info(f"Loading Qwen3 TTS model on {DEVICE} with dtype {dtype}...")
        MODEL = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            device_map=DEVICE,
            dtype=dtype,
            attn_implementation="eager",
        )
        logging.info("✅ Qwen3 TTS model loaded")
    return MODEL



class AudioSpeechIn(BaseModel):
    input: str = Field(..., description="Text to synthesize")
    instructions: str = Field(..., description="Instruction to follow")

@router.post("/tts/audio/speech")
async def audio_speech(body: AudioSpeechIn):
    try:
        text = (body.input or "").strip()
        instruct = (body.instructions or "").strip()

        model = get_model()
        
        wavs, sr = model.generate_voice_design(
            text=text,
            language="English",
            instruct=instruct,
        )

        if isinstance(wavs, torch.Tensor):
            wavs = wavs.cpu().numpy()
        
        # If wavs is a list of arrays, concatenate them
        if isinstance(wavs, list):
            wavs = np.concatenate(wavs, axis=-1)
            
        # Ensure it's 1D (Mono) for simple WAV writing
        wavs = wavs.flatten()

        # 3. Write to in-memory buffer
        buffer = io.BytesIO()
        sf.write(buffer, wavs, sr, format='WAV')
        buffer.seek(0)

        return StreamingResponse(
            buffer, 
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=e)

requirements.txt-
fastapi
uvicorn
qwen-tts
soundfile
numpy
websockets

#Dockerfile-
FROM python:3.12-slim

ENV http_proxy="http://163.116.128.80:8080"
ENV https_proxy="http://163.116.128.80:8080"

RUN apt-get update && apt-get install -y

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py /app
COPY router.py /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

#main.py-
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qwen_tts import Qwen3TTSModel

from router import router


app = FastAPI()

app.include_router(router)

why is this working because this is also using openai

it's client is also similar 
from openai import OpenAI

client = OpenAI(
    base_url="https://qwen-tts-openai-no-docker-150916788856.europe-west1.run.app/v1/tts",
    #base_url="http://127.0.0.1:8080/v1/tts",
    api_key="not-needed",
)

with client.audio.speech.with_streaming_response.create(
    model="",
    voice="",
    input="Hello world from qwen openapi deployed on gcp without docker",
    instructions="speak in a friendly female voice",
) as resp:
    resp.stream_to_file("output.mp3")

print("Saved -> output.mp3")

