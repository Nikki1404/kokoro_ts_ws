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


this is websocket one 
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG http_proxy
ARG https_proxy

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}

RUN apt-get update && apt-get install -y \
        software-properties-common \
        curl \
        ffmpeg \
        build-essential \
        gcc \
        g++ \
        make \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
        python3.10 \
        python3.10-venv \
        python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV KOKORO_LANG=a
ENV KOKORO_DEFAULT_VOICE=af_heart

WORKDIR /app

COPY ./src /app/

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 4000

CMD ["python3", "ws_kokoro_server.py", "--host", "0.0.0.0", "--port", "4000"]
                                                                                                                                        
