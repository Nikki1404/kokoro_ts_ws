 => [7/9] RUN pip3 install -r server_requirements.txt                                                                         372.6s
 => ERROR [8/9] RUN python3 -c "import torch; torch.hub.load('snakers4/silero-vad','silero_vad',trust_repo=True); print('Siler  7.0s
------
 > [8/9] RUN python3 -c "import torch; torch.hub.load('snakers4/silero-vad','silero_vad',trust_repo=True); print('Silero cached')":
4.562 Traceback (most recent call last):
4.562   File "/usr/local/lib/python3.10/dist-packages/torch/_ops.py", line 1442, in load_library
4.562     ctypes.CDLL(path)
4.562   File "/usr/lib/python3.10/ctypes/__init__.py", line 374, in __init__
4.563     self._handle = _dlopen(self._name, mode)
4.563 OSError: /usr/local/lib/python3.10/dist-packages/torchaudio/lib/libtorchaudio.so: undefined symbol: _ZNK5torch8autograd4Node4nameEv
4.563
4.563 The above exception was the direct cause of the following exception:
4.563
4.563 Traceback (most recent call last):
4.563   File "<string>", line 1, in <module>
4.563   File "/usr/local/lib/python3.10/dist-packages/torch/hub.py", line 656, in load
4.563     model = _load_local(repo_or_dir, model, *args, **kwargs)
4.563   File "/usr/local/lib/python3.10/dist-packages/torch/hub.py", line 686, in _load_local
4.564     hub_module = _import_module(MODULE_HUBCONF, hubconf_path)
4.564   File "/usr/local/lib/python3.10/dist-packages/torch/hub.py", line 115, in _import_module
4.564     spec.loader.exec_module(module)
4.564   File "<frozen importlib._bootstrap_external>", line 883, in exec_module
4.564   File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
4.564   File "/root/.cache/torch/hub/snakers4_silero-vad_master/hubconf.py", line 6, in <module>
4.564     from silero_vad.utils_vad import (init_jit_model,
4.564   File "/root/.cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/__init__.py", line 7, in <module>
4.564     from silero_vad.model import load_silero_vad
4.564   File "/root/.cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/model.py", line 1, in <module>
4.564     from .utils_vad import init_jit_model, OnnxWrapper
4.565   File "/root/.cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/utils_vad.py", line 2, in <module>
4.565     import torchaudio
4.565   File "/usr/local/lib/python3.10/dist-packages/torchaudio/__init__.py", line 1, in <module>
4.565     from . import (  # noqa: F401
4.565   File "/usr/local/lib/python3.10/dist-packages/torchaudio/_extension/__init__.py", line 45, in <module>
4.566     _load_lib("libtorchaudio")
4.566   File "/usr/local/lib/python3.10/dist-packages/torchaudio/_extension/utils.py", line 64, in _load_lib
4.567     torch.ops.load_library(path)
4.567   File "/usr/local/lib/python3.10/dist-packages/torch/_ops.py", line 1444, in load_library
4.567     raise OSError(f"Could not load this library: {path}") from e
4.567 OSError: Could not load this library: /usr/local/lib/python3.10/dist-packages/torchaudio/lib/libtorchaudio.so
4.572 Downloading: "https://github.com/snakers4/silero-vad/zipball/master" to /root/.cache/torch/hub/master.zip
------
ERROR: failed to build: failed to solve: process "/bin/sh -c python3 -c \"import torch; torch.hub.load('snakers4/silero-vad','silero_vad',trust_repo=True); print('Silero cached')\"" did not complete successfully: exit code: 1
