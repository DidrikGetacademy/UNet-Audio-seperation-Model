import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import stempeg  # Ensure stempeg is imported
from Training.Externals.Logger import setup_logger

dataset_logger = setup_logger(
    'dataset_Musdb18',
    r'C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_performance_logg\log\Model_Training_logg.txt'
)

class DSD100(Dataset):
    def __init__(self, root_dir, subset='Dev', sr=44100, n_fft=2048, hop_length=512, max_length_seconds=5, max_files=None):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_files = max_files
        self.max_length_seconds = max_length_seconds

        # Directories for mixtures and sources
        self.mixtures_dir = os.path.join(root_dir, 'Mixtures', subset)
        self.sources_dir = os.path.join(root_dir, 'Sources', subset)

        # Get song folders (common between mixtures and sources)
        self.song_folders = [
            song for song in os.listdir(self.mixtures_dir)
            if os.path.isdir(os.path.join(self.mixtures_dir, song))
        ]

    def __len__(self):
        return len(self.song_folders)

    def __getitem__(self, idx):
        song_folder = self.song_folders[idx]

        # Paths for mixture and vocals
        mixture_path = os.path.join(self.mixtures_dir, song_folder, 'mixture.wav')
        vocals_path = os.path.join(self.sources_dir, song_folder, 'vocals.wav')

        # Load mixture and vocals
        mixture, _ = librosa.load(mixture_path, sr=self.sr, mono=True)
        vocals, _ = librosa.load(vocals_path, sr=self.sr, mono=True)

        # Process mixture
        mixture = self._pad_or_trim(mixture)
        mixture = self._normalize(mixture)
        mixture_stft = librosa.stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length)
        mixture_mag = self._adjust_length(np.abs(mixture_stft))

        # Process vocals
        vocals = self._pad_or_trim(vocals)
        vocals = self._normalize(vocals)
        vocals_stft = librosa.stft(vocals, n_fft=self.n_fft, hop_length=self.hop_length)
        vocals_mag = self._adjust_length(np.abs(vocals_stft))

        return (
            torch.tensor(mixture_mag, dtype=torch.float32).unsqueeze(0),
            torch.tensor(vocals_mag, dtype=torch.float32).unsqueeze(0),
        )

    def _pad_or_trim(self, audio):
        max_length_samples = int(self.sr * self.max_length_seconds)
        if len(audio) < max_length_samples:
            return np.pad(audio, (0, max_length_samples - len(audio)), mode='constant')
        return audio[:max_length_samples]

    def _normalize(self, audio):
        return audio / (np.max(np.abs(audio)) + 1e-8)

    def _adjust_length(self, spectrogram):
        desired_time_dim = (self.max_length_seconds * self.sr) // self.hop_length
        if spectrogram.shape[1] < desired_time_dim:
            return np.pad(spectrogram, ((0, 0), (0, desired_time_dim - spectrogram.shape[1])), mode='constant')
        return spectrogram[:, :desired_time_dim]
