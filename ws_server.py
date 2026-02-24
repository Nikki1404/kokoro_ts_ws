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
