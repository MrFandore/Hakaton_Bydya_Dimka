import torch
import torchaudio
import librosa
import numpy as np
import noisereduce as nr
import os


def process_audio(input_path):
    waveform, sample_rate = librosa.load(input_path, sr=16000)

    waveform_tensor = torch.from_numpy(waveform).float()

    reduced_noise = nr.reduce_noise(
        y=waveform.numpy(),
        sr=sample_rate,
        stationary=True
    )

    processed_audio = torchaudio.functional.highpass_biquad(
        waveform_tensor.unsqueeze(0),
        sample_rate,
        cutoff_freq=300
    )

    output_path = input_path.replace('input', 'processed')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, processed_audio, sample_rate)

    return output_path
