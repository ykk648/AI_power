# -- coding: utf-8 --
# @Time : 2023/5/25
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
"""

svc infer -c logs/44k/config.json -m logs/44k/G_2400.pth "*.wav"
"""

import json
import os
import subprocess
from pathlib import Path

# import gradio as gr
import librosa
import numpy as np
import torch
from demucs.apply import apply_model
from demucs.pretrained import DEFAULT_MODEL, get_model
from huggingface_hub import hf_hub_download, list_repo_files
import soundfile as sf

from so_vits_svc_fork.hparams import HParams
from so_vits_svc_fork.inference.core import Svc

# Limit on duration of audio at inference time. increase if you can
# In this parent app, we set the limit with an env var to 30 seconds
# If you didnt set env var + you go OOM try changing 9e9 to <=300ish
duration_limit = int(os.environ.get("MAX_DURATION_SECONDS", 9e9))


class SoVits:
    def __init__(self, generator_path, config_path, cluster_model_path):
        hparams = HParams(**json.loads(Path(config_path).read_text()))
        self.speaker = list(hparams.spk.keys())[0]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Svc(net_g_path=generator_path, config_path=config_path, device=device,
                         cluster_model_path=cluster_model_path)

    def forward(self,
                audio,
                output_path,
                transpose: int = 0,
                auto_predict_f0: bool = False,
                cluster_infer_ratio: float = 0,
                noise_scale: float = 0.4,
                f0_method: str = "crepe",
                db_thresh: int = -40,
                pad_seconds: float = 0.5,
                chunk_seconds: float = 0.5,
                absolute_thresh: bool = False,
                ):
        audio, _ = librosa.load(audio, sr=self.model.target_sample, duration=duration_limit)
        audio = self.model.infer_silence(
            audio.astype(np.float32),
            speaker=self.speaker,
            transpose=transpose,
            auto_predict_f0=auto_predict_f0,
            cluster_infer_ratio=cluster_infer_ratio,
            noise_scale=noise_scale,
            f0_method=f0_method,
            db_thresh=db_thresh,
            pad_seconds=pad_seconds,
            chunk_seconds=chunk_seconds,
            absolute_thresh=absolute_thresh,
        )

        sf.write(output_path, audio, self.model.target_sample, 'PCM_24')
        return audio


if __name__ == "__main__":
    generator_path = './G_329600.pth'
    config_path = "./config.json"
    cluster_model_path = None
    sv = SoVits(generator_path, config_path, cluster_model_path)

    input_path = 'test.wav'

    # output_path = input_path.replace('.wav', '_swap.wav')
    # _ = sv.forward(input_path, output_path, auto_predict_f0=False)

    for f0_predict_method in ['crepe', 'parselmouth', 'dio', 'harvest']:
        output_path = input_path.replace('.wav', f'_swap_auto_predict_{f0_predict_method}.wav')
        _ = sv.forward(input_path, output_path, auto_predict_f0=True, f0_method=f0_predict_method)

    for i in range(-2, 2, 4):
        output_path = input_path.replace('.wav', f'_swap_t_{i}.wav')
        _ = sv.forward(input_path, output_path, auto_predict_f0=False, transpose=i)
