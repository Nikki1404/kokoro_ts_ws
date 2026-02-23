docker build \
  --build-arg http_proxy=http://163.116.128.80:8080 \
  --build-arg https_proxy=http://163.116.128.80:8080 \
  -t kokoro_openai_tts .

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
