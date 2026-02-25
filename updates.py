RUN pip install torch==2.1.2+cu121 torchaudio==2.1.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 🔥 STEP 2: Install other server deps EXCEPT torch
COPY requirements-server.txt .
RUN pip install --no-deps -r requirements-server.txt

# 🔥 STEP 3: Install NeMo without dependencies
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
