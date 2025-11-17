import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class VCTKDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.samples = []
        self.speaker_to_samples = defaultdict(list)

        # Собираем все аудиофайлы и группируем по дикторам
        for speaker_id in os.listdir(os.path.join(data_path, 'wav48')):
            speaker_dir = os.path.join(data_path, 'wav48', speaker_id)
            if not os.path.isdir(speaker_dir):
                continue
            for wav_file in os.listdir(speaker_dir):
                if wav_file.endswith('.wav'):
                    file_path = os.path.join(speaker_dir, wav_file)
                    self.samples.append((file_path, speaker_id))
                    self.speaker_to_samples[speaker_id].append(file_path)
        
        self.speaker_ids = list(self.speaker_to_samples.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_path, anchor_speaker_id = self.samples[idx]

        # Выбираем Positive: другой файл того же диктора
        positive_path = anchor_path
        while positive_path == anchor_path:
            positive_path = random.choice(self.speaker_to_samples[anchor_speaker_id])
        
        # Выбираем Negative: файл другого диктора
        negative_speaker_id = anchor_speaker_id
        while negative_speaker_id == anchor_speaker_id:
            negative_speaker_id = random.choice(self.speaker_ids)
        negative_path = random.choice(self.speaker_to_samples[negative_speaker_id])

        # Загружаем и преобразуем аудио
        anchor_wav, sr = torchaudio.load(anchor_path)
        positive_wav, _ = torchaudio.load(positive_path)
        negative_wav, _ = torchaudio.load(negative_path)
        
        # Применяем трансформацию (извлечение мел-спектрограммы)
        if self.transform:
            # Приводим к фиксированной длине, чтобы спектрограммы были одного размера
            # Например, 3 секунды аудио (48000 * 3 = 144000)
            target_len = 48000 * 3
            anchor_wav = self._pad_or_truncate(anchor_wav, target_len)
            positive_wav = self._pad_or_truncate(positive_wav, target_len)
            negative_wav = self._pad_or_truncate(negative_wav, target_len)
            
            anchor_spec = self.transform(anchor_wav)
            positive_spec = self.transform(positive_wav)
            negative_spec = self.transform(negative_wav)

        return anchor_spec, positive_spec, negative_spec

    def _pad_or_truncate(self, wav, target_len):
        if wav.shape[1] > target_len:
            return wav[:, :target_len]
        else:
            padding = target_len - wav.shape[1]
            return torch.nn.functional.pad(wav, (0, padding))

# Определяем трансформацию для извлечения мел-спектрограмм
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=48000,
    n_fft=2048,
    hop_length=512,
    n_mels=128
)
