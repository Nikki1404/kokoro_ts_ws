 => ERROR [ 9/10] RUN python3 -c "import torch, torchaudio; torch.hub.load('snakers4/silero-vad','silero_vad',trust_repo=True); print('Silero cached'  4.6s
------
 > [ 9/10] RUN python3 -c "import torch, torchaudio; torch.hub.load('snakers4/silero-vad','silero_vad',trust_repo=True); print('Silero cached')":
3.232 Traceback (most recent call last):
3.232   File "/usr/local/lib/python3.10/dist-packages/torch/_ops.py", line 1442, in load_library
3.232     ctypes.CDLL(path)
3.232   File "/usr/lib/python3.10/ctypes/__init__.py", line 374, in __init__
3.233     self._handle = _dlopen(self._name, mode)
3.233 OSError: /usr/local/lib/python3.10/dist-packages/torchaudio/lib/libtorchaudio.so: undefined symbol: _ZNK5torch8autograd4Node4nameEv
3.233
3.233 The above exception was the direct cause of the following exception:
3.233
3.233 Traceback (most recent call last):
3.233   File "<string>", line 1, in <module>
3.233   File "/usr/local/lib/python3.10/dist-packages/torchaudio/__init__.py", line 2, in <module>
3.234     from . import _extension  # noqa  # usort: skip
3.234   File "/usr/local/lib/python3.10/dist-packages/torchaudio/_extension/__init__.py", line 38, in <module>
3.235     _load_lib("libtorchaudio")
3.235   File "/usr/local/lib/python3.10/dist-packages/torchaudio/_extension/utils.py", line 60, in _load_lib
3.235     torch.ops.load_library(path)
3.235   File "/usr/local/lib/python3.10/dist-packages/torch/_ops.py", line 1444, in load_library
3.236     raise OSError(f"Could not load this library: {path}") from e
3.236 OSError: Could not load this library: /usr/local/lib/python3.10/dist-packages/torchaudio/lib/libtorchaudio.so
------
Dockerfile:29
--------------------
  27 |     RUN python3 -m pip install --no-cache-dir -r server_requirements.txt
  28 |
  29 | >>> RUN python3 -c "import torch, torchaudio; torch.hub.load('snakers4/silero-vad','silero_vad',trust_repo=True); print('Silero cached')"
  30 |
  31 |     COPY server.py .
--------------------
ERROR: failed to build: failed to solve: process "/bin/sh -c python3 -c \"import torch, torchaudio; torch.hub.load('snakers4/silero-vad','silero_vad',trust_repo=True); print('Silero cached')\"" did not complete successfully: exit code: 1
