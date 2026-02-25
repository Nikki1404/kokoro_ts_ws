 => [7/9] RUN pip3 install -r server_requirements.txt                                                                         377.8s
 => ERROR [8/9] RUN python3 -c "import torch; torch.hub.load('snakers4/silero-vad','silero_vad',trust_repo=True); print('Siler  6.3s
------
 > [8/9] RUN python3 -c "import torch; torch.hub.load('snakers4/silero-vad','silero_vad',trust_repo=True); print('Silero cached')":
4.602 Traceback (most recent call last):
4.602   File "<string>", line 1, in <module>
4.602   File "/usr/local/lib/python3.10/dist-packages/torch/hub.py", line 656, in load
4.602     model = _load_local(repo_or_dir, model, *args, **kwargs)
4.602   File "/usr/local/lib/python3.10/dist-packages/torch/hub.py", line 686, in _load_local
4.603     hub_module = _import_module(MODULE_HUBCONF, hubconf_path)
4.603   File "/usr/local/lib/python3.10/dist-packages/torch/hub.py", line 115, in _import_module
4.603     spec.loader.exec_module(module)
4.603   File "<frozen importlib._bootstrap_external>", line 883, in exec_module
4.603   File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
4.603   File "/root/.cache/torch/hub/snakers4_silero-vad_master/hubconf.py", line 6, in <module>
4.603     from silero_vad.utils_vad import (init_jit_model,
4.603   File "/root/.cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/__init__.py", line 7, in <module>
4.603     from silero_vad.model import load_silero_vad
4.603   File "/root/.cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/model.py", line 1, in <module>
4.604     from .utils_vad import init_jit_model, OnnxWrapper
4.604   File "/root/.cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/utils_vad.py", line 2, in <module>
4.604     import torchaudio
4.604 ModuleNotFoundError: No module named 'torchaudio'
4.609 Downloading: "https://github.com/snakers4/silero-vad/zipball/master" to /root/.cache/torch/hub/master.zip
------
ERROR: failed to build: failed to solve: process "/bin/sh -c python3 -c \"import torch; torch.hub.load('snakers4/silero-vad','silero_vad',trust_repo=True); print('Silero cached')\"" did not complete successfully: exit code: 1
