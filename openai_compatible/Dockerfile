FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG http_proxy
ARG https_proxy

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

ENV KOKORO_LANG=a
ENV KOKORO_DEFAULT_VOICE=af_heart
ENV KOKORO_PRELOAD_VOICES="af_heart af_bella af_sky"
ENV PRELOAD_ON_STARTUP=1
ENV LOG_LEVEL=INFO

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/

EXPOSE 8081

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081"]
