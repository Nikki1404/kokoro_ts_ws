FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential gcc g++ make \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /app/requirements.txt

COPY . /app/

RUN pip install --no-cache-dir uvicorn

ENV KOKORO_LANG=a
EXPOSE 8080

CMD ["uvicorn", "ws_kokoro_server:app", "--host", "0.0.0.0", "--port", "8080"]
