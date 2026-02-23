FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python config
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Kokoro defaults
ENV KOKORO_LANG=a
ENV KOKORO_DEFAULT_VOICE=af_heart
ENV KOKORO_PRELOAD_VOICES="af_heart af_bella af_sky"

# Upgrade pip first
RUN python3 -m pip install --upgrade pip

# Install core Python deps (excluding kokoro-tts)
COPY requirements.txt /app/
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# 🔥 Install REAL Kokoro from GitHub (NOT kokoro-tts)
RUN python3 -m pip install --no-cache-dir \
    git+https://github.com/nvidia/kokoro.git

# Install GPU runtime explicitly
RUN python3 -m pip install --no-cache-dir onnxruntime-gpu

# Copy project
COPY . /app/

EXPOSE 8080

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
