#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio, json, os
import numpy as np
import websockets
import sounddevice as sd
import soundfile as sf

async def tts_once(url, text, voice="af_heart", speed=1.0, fmt="f32"):
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"text": text, "voice": voice, "speed": speed, "format": fmt}))
        sr = 24000
        stream = None
        dtype = np.float32 if fmt == "f32" else np.int16
        audio_buf = bytearray()

        try:
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    # Stream PCM audio
                    if fmt in {"f32", "s16"}:
                        a = np.frombuffer(msg, dtype=dtype)
                        if fmt == "s16":
                            a = (a.astype(np.float32) / 32767.0)
                        if stream is None:
                            stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
                            stream.start()
                        stream.write(a.reshape(-1, 1))
                    else:
                        # For encoded audio (mp3, wav, etc.)
                        audio_buf.extend(msg)

                else:
                    data = json.loads(msg)
                    t = data.get("type")
                    if t == "meta":
                        sr = int(data["sample_rate"])
                        print(f"[meta] sr={sr}, fmt={data['sample_format']}")
                    elif t == "ttfa":
                        print(f"[ttfa] {data['ms']:.1f} ms")
                    elif t == "done":
                        if data.get("error"):
                            print("[done:ERROR]", data["error"])
                        else:
                            print(f"[done] gen={data['total_ms']:.1f} ms, "
                                  f"audio={data['audio_ms']:.1f} ms, "
                                  f"segments={data['segments']}, rtf={data['rtf']:.3f}")
                        break
        finally:
            if stream is not None:
                stream.stop(); stream.close()

            # Save encoded file if needed
            if fmt not in {"f32", "s16"} and len(audio_buf) > 0:
                os.makedirs("out_audio", exist_ok=True)
                out_path = f"out_audio/output_{fmt}.{'mp3' if fmt == 'mp3' else fmt}"
                with open(out_path, "wb") as f:
                    f.write(audio_buf)
                print(f"[saved] {out_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8080/ws")
    ap.add_argument("--text", default=None)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--fmt", default="f32", choices=["f32", "s16", "wav", "mp3", "ogg", "flac"])
    args = ap.parse_args()

    async def run():
        if args.text:
            await tts_once(args.url, args.text, args.voice, args.speed, args.fmt)
            return

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
