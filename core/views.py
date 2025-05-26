import os
import librosa
import noisereduce as nr
import numpy as np
import torchaudio
from django.shortcuts import render
from django.conf import settings
from .forms import UploadAudioForm
from .utils.transcription import transcribe_audio
from .utils.classification import classify_war_song
import torch

def enhance_audio(input_path, output_path):
    y, sr = librosa.load(input_path, sr=None)
    reduced_noise = nr.reduce_noise(y=y, sr=sr, stationary=True)
    y_filtered = librosa.effects.preemphasis(reduced_noise, coef=0.97)
    y_enhanced = librosa.effects.trim(y_filtered, top_db=20)[0]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, torch.from_numpy(y_enhanced).unsqueeze(0), sr)


def upload_audio(request):
    form = UploadAudioForm()
    context = {'show_result': False}

    if request.method == 'POST':
        form = UploadAudioForm(request.POST, request.FILES)
        if form.is_valid():
            audio_file = form.cleaned_data['audio']
            input_dir = os.path.join(settings.MEDIA_ROOT, 'input')
            os.makedirs(input_dir, exist_ok=True)
            input_path = os.path.join(input_dir, audio_file.name)

            with open(input_path, 'wb+') as f:
                for chunk in audio_file.chunks():
                    f.write(chunk)

            processed_dir = os.path.join(settings.MEDIA_ROOT, 'processed')
            processed_path = os.path.join(processed_dir, audio_file.name)
            enhance_audio(input_path, processed_path)
            transcribed_text = transcribe_audio(processed_path)
            if not transcribed_text.strip():
                transcribed_text = "Текст не распознан"

            try:
                analysis_result = classify_war_song(transcribed_text)
            except Exception as e:
                analysis_result = {
                    'year': 'не определен',
                    'themes': ['ошибка анализа'],
                    'authors': []
                }

            context.update({
                'show_result': True,
                'original_audio': f'/media/input/{audio_file.name}',
                'processed_audio': f'/media/processed/{audio_file.name}',
                'text': transcribed_text,
                **analysis_result
            })

    context['form'] = form
    return render(request, 'core/result.html', context)
