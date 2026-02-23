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
