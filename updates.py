FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ARG http_proxy
ARG https_proxy

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV KOKORO_LANG=a
ENV KOKORO_DEFAULT_VOICE=af_heart
ENV KOKORO_PRELOAD_VOICES="af_heart af_bella af_sky"

RUN python3 -m pip install --upgrade pip

COPY requirements.txt /app/
RUN python3 -m pip install --no-cache-dir -r requirements.txt

RUN python3 -m pip install --no-cache-dir \
    git+https://github.com/nvidia/kokoro.git

RUN python3 -m pip install --no-cache-dir onnxruntime-gpu
COPY . /app/

EXPOSE 8081

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8081"]
