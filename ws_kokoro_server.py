#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, time, os, traceback, logging
from typing import List
from io import BytesIO
import numpy as np
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

logging.basicConfig(level=logging.INFO)

SR = 24000
app = FastAPI()
pipeline = None


def as_numpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    return x.astype(np.float32).reshape(-1)


@app.on_event("startup")
async def _init():
    global pipeline
    pipeline = KPipeline(lang_code=os.getenv("KOKORO_LANG", "a"))


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
    try:
        while True:   # 🔥 allow multiple messages in same connection
            req = await websocket.receive_text()
            cfg = json.loads(req)

            text = str(cfg.get("text", "")).strip()
            voice = str(cfg.get("voice", "af_heart"))
            speed = float(cfg.get("speed", 1.0))
            fmt = str(cfg.get("format", "f32")).lower()

            if not text:
                await websocket.send_text(json.dumps({"type": "done", "error": "empty text"}))
                continue   # don’t close, wait for next message

            await websocket.send_text(json.dumps({
                "type": "meta",
                "sample_rate": SR,
                "channels": 1,
                "sample_format": fmt
            }))

            t0 = time.perf_counter()
            ttfa_sent = False
            segments = 0
            audio_total_s = 0.0
            buf: List[np.ndarray] = []

            gen = pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+")
            for (_gs, _ps, audio) in gen:
                now = time.perf_counter()
                if not ttfa_sent:
                    await websocket.send_text(json.dumps({"type": "ttfa", "ms": (now - t0) * 1000.0}))
                    ttfa_sent = True

                a = as_numpy(audio)
                buf.append(a)
                segments += 1
                audio_total_s += a.size / SR

                # Stream only for PCM
                if fmt in {"f32", "s16"}:
                    if fmt == "s16":
                        pcm = (np.clip(a, -1, 1) * 32767.0).astype(np.int16).tobytes()
                    else:
                        pcm = a.tobytes()
                    await websocket.send_bytes(pcm)

            total_ms = (time.perf_counter() - t0) * 1000.0
            rtf = (total_ms / 1000.0) / max(1e-6, audio_total_s)

            # Encode full output for compressed formats
            if fmt in {"wav", "mp3", "ogg", "flac"}:
                full_audio = np.concatenate(buf) if len(buf) > 1 else buf[0]
                wav_io = BytesIO()
                sf.write(wav_io, np.clip(full_audio, -1, 1), SR, format="WAV", subtype="PCM_16")
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
                "error": None
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


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=False)
