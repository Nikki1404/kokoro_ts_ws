RUN pip install torch==2.1.2+cu121 torchaudio==2.1.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements-server.txt .
RUN pip install --no-deps -r requirements-server.txt
RUN pip install nemo_toolkit[asr] --no-deps

RUN pip install \
    hydra-core \
    omegaconf \
    pytorch-lightning \
    librosa \
    soundfile \
    einops \
    braceexpand \
    editdistance \
    wget \
    sox \
    scipy

RUN python3 -c "import torch, torchaudio; torch.hub.load('snakers4/silero-vad','silero_vad',trust_repo=True); print('Silero cached')"

COPY server.py .
