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


def as_numpy(x):
    """Convert tensors to numpy arrays."""
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    return x.astype(np.float32).reshape(-1)


@app.on_event("startup")
async def _init():
    """Initialize Kokoro TTS pipeline."""
    global pipeline
    pipeline = KPipeline(lang_code=CONFIG["lang_code"])
    logging.info(f"✅ Kokoro pipeline initialized (lang={CONFIG['lang_code']})")


@app.get("/", response_class=PlainTextResponse)
def root():
    return (
        "Kokoro TTS WebSocket server.\n"
        "Connect to /ws with JSON: {\"text\",\"voice\",\"speed\",\"format\"}\n"
        "format: f32 | s16 | wav | mp3 | ogg | flac\n"
    )


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    logging.info("connection open")
    try:
        while True:  
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

            start_time = time.time()
            start_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            logging.info(f"[start] {start_human}")

            t0 = time.perf_counter()
            ttfa_sent = False
            segments = 0
            audio_total_s = 0.0
            buf: List[np.ndarray] = []

            gen = pipeline(text, voice=voice, speed=speed,
                           split_pattern=CONFIG["pipeline"]["split_pattern"])
            for (_gs, _ps, audio) in gen:
                now = time.perf_counter()
                if not ttfa_sent:
                    await websocket.send_text(json.dumps(
                        {"type": "ttfa", "ms": (now - t0) * 1000.0}))
                    ttfa_sent = True

                a = as_numpy(audio)
                buf.append(a)
                segments += 1
                audio_total_s += a.size / SR

                if fmt in {"f32", "s16"}:
                    if fmt == "s16":
                        pcm = (np.clip(a, -1, 1) * 32767.0).astype(np.int16).tobytes()
                    else:
                        pcm = a.tobytes()
                    await websocket.send_bytes(pcm)

            total_ms = (time.perf_counter() - t0) * 1000.0
            rtf = (total_ms / 1000.0) / max(1e-6, audio_total_s)

            end_time = time.time()
            end_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            total_time_sec = end_time - start_time 

            logging.info(f"[end]   {end_human}")
            logging.info(f"[total] {total_time_sec:.9f} sec")  # full microsecond precision

            if fmt in {"wav", "mp3", "ogg", "flac"}:
                full_audio = np.concatenate(buf) if len(buf) > 1 else buf[0]
                wav_io = BytesIO()
                sf.write(wav_io, np.clip(full_audio, -1, 1), SR,
                         format="WAV", subtype="PCM_16")
                wav_io.seek(0)
                audio_seg = AudioSegment.from_wav(wav_io)
                out_io = BytesIO()
                audio_seg.export(out_io, format=fmt)
                await websocket.send_bytes(out_io.getvalue())

            await websocket.send_text(json.dumps({
                "type": "done",
                "total_ms": total_ms,
                "audio_ms": audio_total_s * 1000.0,
                "segments": segments,
                "rtf": rtf,
                "error": None,
                "start_time": start_time,
                "end_time": end_time,
                "total_time_sec": total_time_sec
            }))

    except WebSocketDisconnect:
        logging.warning("Client disconnected.")
    except Exception as e:
        logging.error(traceback.format_exc())
        try:
            await websocket.send_text(json.dumps({"type": "done", "error": str(e)}))
        except Exception:
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
