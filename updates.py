RUN pip install nemo_toolkit[asr] --no-deps

# 🔥 Install NeMo required extras manually (safe ones)
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

# 🔥 Prefetch Silero
RUN python3 -c "import torch, torchaudio; torch.hub.load('snakers4/silero-vad','silero_vad',trust_repo=True); print('Silero cached')"

COPY server.py .
