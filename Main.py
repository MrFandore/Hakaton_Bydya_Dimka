import os
import asyncio
import concurrent.futures
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
import torch
import whisper
from transformers import pipeline
import noisereduce as nr
from scipy.signal import butter, filtfilt, wiener, iirnotch, medfilt, correlate
from datetime import datetime
import logging
from demucs.pretrained import get_model
from demucs.apply import apply_model
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*", category=UserWarning,
    module="torch.nn.modules.activation"
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Фильтры ===
def bandpass_filter(data, sr, lowcut=80, highcut=14500):
    b, a = butter(6, [lowcut / (0.5 * sr), highcut / (0.5 * sr)], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data, sr, freq=50.0, Q=30):
    b, a = iirnotch(freq / (sr / 2), Q)
    return filtfilt(b, a, data)

def highpass_filter(data, sr, cutoff=80, order=4):
    b, a = butter(order, cutoff / (0.5 * sr), btype='high')
    return filtfilt(b, a, data)

def lowpass_filter(data, sr, cutoff=15000, order=4):
    b, a = butter(order, cutoff / (0.5 * sr), btype='low')
    return filtfilt(b, a, data)

def calculate_snr(clean, noisy):
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - noisy) ** 2)
    if signal_power < 1e-10 or noise_power < 1e-10:
        return 0.0
    return 10 * np.log10(signal_power / noise_power)

def compress(audio, threshold=-20, ratio=4.0):
    db = 20 * np.log10(np.abs(audio) + 1e-8)
    over_threshold = db > threshold
    gain_reduction = over_threshold * (1 - 1/ratio) * (db - threshold)
    new_db = db - gain_reduction
    return np.sign(audio) * (10 ** (new_db / 20))

def align_tracks(vocal, instrumental):
    corr = correlate(instrumental, vocal, mode='full')
    lag = np.argmax(corr) - len(vocal) + 1
    if lag > 0:
        instrumental = instrumental[lag:]
        vocal = vocal[:len(instrumental)]
    else:
        vocal = vocal[-lag:]
        instrumental = instrumental[:len(vocal)]
    return vocal, instrumental

def save_parallel_mix(vocal, instrumental, sr, path):
    min_len = min(vocal.shape[-1], instrumental.shape[-1])
    vocal = vocal[0][:min_len]
    instr = instrumental[0][:min_len]
    vocal *= 0.8  #усиление вокала
    combined = vocal + instr
    combined = combined / np.max(np.abs(combined)) * 0.95  #громкость
    sf.write(path, combined, sr)


class HistoricalAudioProcessor:
    def __init__(self, output_dir="processed_audio"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Используется устройство: {self.device}")

        self.separator = None
        self.whisper_model = whisper.load_model("medium")
        self.sentiment_analyzer = pipeline(
            "text-classification",
            model="blanchefort/rubert-base-cased-sentiment",
            device=self.device
        )
        self._init_models()

    def _init_models(self):
        try:
            self.separator = get_model(name="htdemucs")
            self.separator.to(self.device)
            logger.info("Модель Demucs загружена успешно")
        except Exception as e:
            logger.error(f"Ошибка: {e}")

    async def process_audio_file(self, input_path):
        input_path = Path(input_path)
        base_name = input_path.stem

        logger.info(f"Начинаем обработку файла: {input_path}")
        audio, sr = librosa.load(input_path, sr=None, mono=False)
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)

        vocal_track, instrumental_track = await self._separate_sources(audio, sr)

        enhanced_vocal = self._process_vocal_enhanced(vocal_track, sr)
        enhanced_instr = self._process_instrumental_enhanced(instrumental_track, sr)

        self._save_parallel_mix(enhanced_vocal, enhanced_instr, sr, base_name)

        await self._process_vocal_analysis(enhanced_vocal, sr, base_name)

        return self.output_dir

    def _compress(self, audio, threshold=-18, ratio=2.0):
        db = 20 * np.log10(np.abs(audio) + 1e-8)
        over_threshold = db > threshold
        gain_reduction = over_threshold * (1 - 1 / ratio) * (db - threshold)
        new_db = db - gain_reduction
        return np.sign(audio) * (10 ** (new_db / 20))

    def _transient_boost(self, audio, factor=1.2):
        from scipy.signal import hilbert
        envelope = np.abs(hilbert(audio))
        boost = (1 + (envelope - np.min(envelope)) / (np.max(envelope) + 1e-6)) * factor
        return audio * boost

    def _process_vocal_enhanced(self, audio, sr):
        audio_mono = audio[0] if audio.ndim > 1 else audio

        #Первый проход
        step1 = nr.reduce_noise(
            y=audio_mono,
            sr=sr,
            stationary=False,
            prop_decrease=0.5,
            time_mask_smooth_ms=80,
            freq_mask_smooth_hz=100
        )

        #Умеренная фильтрация
        step2 = notch_filter(step1, sr)
        step2 = highpass_filter(step2, sr, cutoff=80)  # сохраняем гласные

        #Второй проход NR — мягкая стационарная очистка
        cleaned = nr.reduce_noise(
            y=step2,
            sr=sr,
            stationary=True,
            prop_decrease=0.25,
            time_mask_smooth_ms=50,
            freq_mask_smooth_hz=100
        )

        #Объём в диапазоне (300–700 Гц)
        formant_boost = bandpass_filter(cleaned, sr, lowcut=300, highcut=700)
        formant_boost = np.nan_to_num(formant_boost, nan=0.0)
        cleaned += 0.3 * formant_boost

        #Компрессия (лёгкая)
        cleaned = self._compress(cleaned, threshold=-18, ratio=2.0)

        #Чёткость речи: усиление диапазона согласных
        clarity_boost = bandpass_filter(cleaned, sr, lowcut=1500, highcut=5000)
        cleaned += 0.2 * clarity_boost

        #Усиление переходов для распознавания слогов
        cleaned = self._transient_boost(cleaned, factor=1.1)

        #Снижение высокочастотного шума
        cleaned = lowpass_filter(cleaned, sr, cutoff=12000)

        #Обработка NaN/inf
        if not np.all(np.isfinite(cleaned)):
            logger.warning("Обнаружены некорректные значения (NaN/Inf) в аудиосигнале — применяется автоочистка.")
            cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=1.0, neginf=-1.0)

        #Финальная нормализация с запасом по громкости
        cleaned = librosa.util.normalize(cleaned) * 0.80

        return cleaned.reshape(1, -1)

    def _process_instrumental_enhanced(self, audio, sr):
        audio_mono = audio[0] if audio.ndim > 1 else audio
        audio_mono = lowpass_filter(audio_mono, sr)
        audio_mono = bandpass_filter(audio_mono, sr)
        audio_mono = wiener(audio_mono, mysize=9)
        audio_mono = medfilt(audio_mono, kernel_size=5)
        reduced = nr.reduce_noise(y=audio_mono, sr=sr, stationary=False, prop_decrease=0.7)
        return librosa.util.normalize(reduced).reshape(1, -1)

    def _save_parallel_mix(self, vocal, instrumental, sr, base_name):
        path = self.output_dir / f"{base_name}_parallel.wav"
        save_parallel_mix(vocal, instrumental, sr, path)
        logger.info(f"Сохранён параллельный микс: {path}")

    async def _separate_sources(self, audio, sr):
        logger.info("Разделение источников...")
        if self.separator:
            def separate_with_demucs():
                audio_tensor = torch.from_numpy(audio).float()
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                audio_tensor = audio_tensor.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    sources = apply_model(self.separator, audio_tensor, device=self.device)
                vocal = sources[0, self.separator.sources.index("vocals")].cpu().numpy()
                instrumental = (
                    sources[0, self.separator.sources.index("bass")].cpu().numpy() +
                    sources[0, self.separator.sources.index("drums")].cpu().numpy() +
                    sources[0, self.separator.sources.index("other")].cpu().numpy()
                )
                return vocal, instrumental

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(separate_with_demucs)
                return future.result()
        else:
            return self._simple_source_separation(audio, sr)

    def _simple_source_separation(self, audio, sr):
        stft = librosa.stft(audio[0] if audio.ndim > 1 else audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        vocal_mask = np.zeros_like(magnitude)
        freq_bins = librosa.fft_frequencies(sr=sr)
        mask_indices = (freq_bins >= 300) & (freq_bins <= 3400)
        vocal_mask[mask_indices] = 1.0
        vocal_stft = magnitude * vocal_mask * np.exp(1j * phase)
        instrumental_stft = magnitude * (1 - vocal_mask) * np.exp(1j * phase)
        vocal = librosa.istft(vocal_stft)
        instrumental = librosa.istft(instrumental_stft)
        return vocal.reshape(1, -1), instrumental.reshape(1, -1)

    async def _process_vocal_analysis(self, vocal_audio, sr, base_name):
        logger.info("Начинаем анализ вокальной дорожки...")
        text = await self._extract_text(vocal_audio, sr)
        with open(self.output_dir / f"{base_name}_text.txt", 'w', encoding='utf-8') as f:
            f.write(text)
        metadata = await self._extract_metadata(text)
        with open(self.output_dir / f"{base_name}_metadata.txt", 'w', encoding='utf-8') as f:
            f.write(metadata)
        logger.info("Анализ завершен")

    async def _extract_text(self, audio, sr):
        logger.info("Извлечение текста из аудио...")

        def transcribe():
            audio_mono = audio[0] if audio.ndim > 1 else audio
            audio_mono = audio_mono.astype(np.float32)
            audio_whisper = librosa.resample(audio_mono, orig_sr=sr, target_sr=16000)
            return self.whisper_model.transcribe(audio_whisper, language="ru", task="transcribe")["text"]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            return executor.submit(transcribe).result()

    async def _extract_metadata(self, text):
        logger.info("Анализ метаданных...")

        def analyze():
            metadata = [f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f"Текст песни:\n{text}\n"]
            try:
                sentiment = self.sentiment_analyzer(text[:512])
                metadata.append(f"Настроение: {sentiment[0]['label']} (уверенность: {sentiment[0]['score']:.2f})")
            except Exception as e:
                metadata.append(f"Ошибка анализа настроения: {e}")
            choir = sum(1 for w in ["хор", "вместе", "дружно", "все", "мы"] if w in text.lower())
            solo = sum(1 for w in ["я", "мой", "моя", "мне"] if w in text.lower())
            if choir > solo:
                metadata.append("Тип исполнения: Хоровое исполнение")
            elif solo > choir:
                metadata.append("Тип исполнения: Сольное исполнение")
            else:
                metadata.append("Тип исполнения: Смешанное исполнение")
            war = sum(1 for w in ["война", "враг", "победа", "родина", "фронт", "бой", "солдат", "герой"] if
                      w in text.lower())
            metadata.append(f"Военная тематика: {'Да' if war > 0 else 'Не определена'} (ключевых слов: {war})")
            metadata.append(f"Количество слов: {len(text.split())}")
            metadata.append(f"Количество символов: {len(text)}")
            return "\n".join(metadata)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            return executor.submit(analyze).result()


async def main():
    processor = HistoricalAudioProcessor()
    input_file = "C:/Users/Stan/Desktop/Тест.wav"
    if os.path.exists(input_file):
        result = await processor.process_audio_file(input_file)
        print(f"Обработка завершена: {result}")
    else:
        print("Файл не найден")


if __name__ == "__main__":
    asyncio.run(main())
