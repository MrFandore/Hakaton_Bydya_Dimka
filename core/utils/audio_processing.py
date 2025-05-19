import torch
import torchaudio
import librosa
import numpy as np
import noisereduce as nr
import os


def process_audio(input_path):
    # Загрузка аудио
    waveform, sample_rate = librosa.load(input_path, sr=16000)

    # Конвертация в тензор PyTorch
    waveform_tensor = torch.from_numpy(waveform).float()

    # Шумоподавление
    reduced_noise = nr.reduce_noise(
        y=waveform.numpy(),
        sr=sample_rate,
        stationary=True
    )

    # Улучшение качества
    processed_audio = torchaudio.functional.highpass_biquad(
        waveform_tensor.unsqueeze(0),
        sample_rate,
        cutoff_freq=300
    )

    # Сохранение файла
    output_path = input_path.replace('input', 'processed')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, processed_audio, sample_rate)

    return output_path