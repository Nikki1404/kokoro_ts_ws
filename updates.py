 => ERROR [8/9] RUN python3 -c "import torch; torch.hub.load('snakers4/silero-vad','silero_vad',trust_repo=True); print('Siler  7.0s
------
 > [8/9] RUN python3 -c "import torch; torch.hub.load('snakers4/silero-vad','silero_vad',trust_repo=True); print('Silero cached')":
4.873 Traceback (most recent call last):
4.873   File "/usr/local/lib/python3.10/dist-packages/torch/_ops.py", line 1442, in load_library
4.873     ctypes.CDLL(path)
4.873   File "/usr/lib/python3.10/ctypes/__init__.py", line 374, in __init__
4.874     self._handle = _dlopen(self._name, mode)
4.874 OSError: /usr/local/lib/python3.10/dist-packages/torchaudio/lib/libtorchaudio.so: undefined symbol: _ZNK5torch8autograd4Node4nameEv
4.874
4.874 The above exception was the direct cause of the following exception:
4.874
4.874 Traceback (most recent call last):
4.874   File "<string>", line 1, in <module>
4.874   File "/usr/local/lib/python3.10/dist-packages/torch/hub.py", line 656, in load
4.874     model = _load_local(repo_or_dir, model, *args, **kwargs)
4.874   File "/usr/local/lib/python3.10/dist-packages/torch/hub.py", line 686, in _load_local
4.874     hub_module = _import_module(MODULE_HUBCONF, hubconf_path)
4.874   File "/usr/local/lib/python3.10/dist-packages/torch/hub.py", line 115, in _import_module
4.875     spec.loader.exec_module(module)
4.875   File "<frozen importlib._bootstrap_external>", line 883, in exec_module
4.875   File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
4.875   File "/root/.cache/torch/hub/snakers4_silero-vad_master/hubconf.py", line 6, in <module>
4.875     from silero_vad.utils_vad import (init_jit_model,
4.875   File "/root/.cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/__init__.py", line 7, in <module>
4.875     from silero_vad.model import load_silero_vad
4.875   File "/root/.cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/model.py", line 1, in <module>
4.875     from .utils_vad import init_jit_model, OnnxWrapper
4.875   File "/root/.cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/utils_vad.py", line 2, in <module>
4.875     import torchaudio
4.875   File "/usr/local/lib/python3.10/dist-packages/torchaudio/__init__.py", line 2, in <module>
4.876     from . import _extension  # noqa  # usort: skip
4.876   File "/usr/local/lib/python3.10/dist-packages/torchaudio/_extension/__init__.py", line 38, in <module>
4.877     _load_lib("libtorchaudio")
4.877   File "/usr/local/lib/python3.10/dist-packages/torchaudio/_extension/utils.py", line 60, in _load_lib
4.878     torch.ops.load_library(path)
4.878   File "/usr/local/lib/python3.10/dist-packages/torch/_ops.py", line 1444, in load_library
4.878     raise OSError(f"Could not load this library: {path}") from e
4.878 OSError: Could not load this library: /usr/local/lib/python3.10/dist-packages/torchaudio/lib/libtorchaudio.so
4.883 Downloading: "https://github.com/snakers4/silero-vad/zipball/master" to /root/.cache/torch/hub/master.zip
------
Dockerfile:25
--------------------
  23 |
  24 |     # ---- Prefetch Silero repo into torch hub cache (no runtime download) ----
  25 | >>> RUN python3 -c "import torch; torch.hub.load('snakers4/silero-vad','silero_vad',trust_repo=True); print('Silero cached')"
  26 |
  27 |     COPY server.py .
--------------------
ERROR: failed to build: failed to solve: process "/bin/sh -c python3 -c \"import torch; torch.hub.load('snakers4/silero-vad','silero_vad',trust_repo=True); print('Silero cached')\"" did not complete successfully: exit code: 1
