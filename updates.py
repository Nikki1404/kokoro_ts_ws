after deplying to cloud run 

got this url https://kokoro-ws-150916788856.us-central1.run.app
now how to test it from local via this script 
client.py 
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
    ap.add_argument("--url", default="https://kokoro-ws-150916788856.us-central1.run.app/ws")
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

(kokoro_env) PS C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client> python .\ws_kokoro_client.py
Kokoro WS client (interactive). Type text and press Enter. /q to quit.
text> hi this is tts testing via kokoro ws
[warn] send/play failed: https://kokoro-ws-150916788856.us-central1.run.app/ws isn't a valid URI: scheme isn't ws or wss
