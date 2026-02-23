#requirements.txt-
fastapi
kokoro-tts
numpy
openai
onnxruntime-gpu
pydantic-settings
pydub
requests
sounddevice
uvicorn

#Dockerifle-
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV https_proxy="http://163.116.128.80:8080"
ENV http_proxy="http://163.116.128.80:8080"

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git ffmpeg python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KOKORO_LANG=a \
    KOKORO_DEFAULT_VOICE=af_heart \
    KOKORO_PRELOAD_VOICES="af_heart af_bella af_sky"

COPY . /app/

RUN  pip install --no-cache-dir -r /app/requirements.txt

RUN pip install onnxruntime-gpu

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

I am getting this 

(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/cx-speech-tts/fastapi_impl_gpu# docker run --gpus all -p 8080:8080 -e KOKORO_API_KEY=devkey kokoro_openai_tts

==========
== CUDA ==
==========

CUDA Version 12.1.1

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

Traceback (most recent call last):
  File "/app/app/tts/kokoro_engine.py", line 10, in <module>
    from kokoro import KPipeline
ModuleNotFoundError: No module named 'kokoro'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/bin/uvicorn", line 8, in <module>
    sys.exit(main())
  File "/usr/local/lib/python3.10/dist-packages/click/core.py", line 1485, in __call__
    return self.main(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/click/core.py", line 1406, in main
    rv = self.invoke(ctx)
  File "/usr/local/lib/python3.10/dist-packages/click/core.py", line 1269, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "/usr/local/lib/python3.10/dist-packages/click/core.py", line 824, in invoke
    return callback(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/main.py", line 433, in main
    run(
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/main.py", line 606, in run
    server.run()
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/server.py", line 75, in run
    return asyncio_run(self.serve(sockets=sockets), loop_factory=self.config.get_loop_factory())
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/_compat.py", line 60, in asyncio_run
    return loop.run_until_complete(main)
  File "/usr/lib/python3.10/asyncio/base_events.py", line 649, in run_until_complete
    return future.result()
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/server.py", line 79, in serve
    await self._serve(sockets)
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/server.py", line 86, in _serve
    config.load()
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/config.py", line 441, in load
    self.loaded_app = import_from_string(self.app)
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/importer.py", line 19, in import_from_string
    module = importlib.import_module(module_str)
  File "/usr/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1050, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1027, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1006, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 688, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 883, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/app/app/main.py", line 4, in <module>
    from app.routers.openai_compatible import router as openai_router
  File "/app/app/routers/openai_compatible.py", line 8, in <module>
    from app.tts.kokoro_engine import synthesize_np, encode_audio, maybe_save
  File "/app/app/tts/kokoro_engine.py", line 12, in <module>
    from kokoro_tts import KPipeline
  File "/usr/local/lib/python3.10/dist-packages/kokoro_tts/__init__.py", line 20, in <module>
    import sounddevice as sd
  File "/usr/local/lib/python3.10/dist-packages/sounddevice.py", line 72, in <module>
    raise OSError('PortAudio library not found')
OSError: PortAudio library not found
