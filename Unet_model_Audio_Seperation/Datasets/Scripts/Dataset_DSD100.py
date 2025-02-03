# dataset1.py
import os
import os
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Training.Externals.Logger import setup_logger
from Training.Externals.utils import Return_root_dir
root_dir = Return_root_dir()  # Gets the root directory
train_log_path = os.path.join(root_dir, "Model_Performance_logg/log/datasets.txt")
dataset_logger = setup_logger('dataset_Musdb18', train_log_path)

class DSD100(Dataset):
    def __init__(
        self, 
        root_dir,
        subset='Dev',
        sr=44100,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=15,
        max_files=None
        ):
        
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length_seconds = max_length_seconds
        self.max_files = max_files

        self.mixtures_dir = os.path.join(root_dir, 'Mixtures', subset)
        self.sources_dir = os.path.join(root_dir, 'Sources', subset)

        self.song_folders = [
            song for song in os.listdir(self.mixtures_dir)
            if os.path.isdir(os.path.join(self.mixtures_dir, song))
        ]
        if max_files is not None:
            self.song_folders = self.song_folders[:max_files]

        # Logging dataset initialization
        dataset_logger.info(f"Initialized DSD100 Dataset with {len(self.song_folders)} song folders")
        dataset_logger.info(f"Sample rate: {self.sr}, N_FFT: {self.n_fft}, Hop Length: {self.hop_length}, Max Length (seconds): {self.max_length_seconds}")

    def __len__(self):
        return len(self.song_folders)

    def __getitem__(self, idx):
        song_folder = self.song_folders[idx]
        mixture_path = os.path.join(self.mixtures_dir, song_folder, 'mixture.wav')
        vocals_path = os.path.join(self.sources_dir, song_folder, 'vocals.wav')

        # Logging file paths
        dataset_logger.debug(f"Processing song folder: {song_folder}")
        dataset_logger.debug(f"Mixture path: {mixture_path}")
        dataset_logger.debug(f"Vocals path: {vocals_path}")

        # Load and process audio
        mixture, _ = librosa.load(mixture_path, sr=self.sr, mono=True)
        vocals, _ = librosa.load(vocals_path, sr=self.sr, mono=True)

        dataset_logger.debug(
            f"[Before Processing] Mixture length: {len(mixture)}, Vocals length: {len(vocals)}"
        )

        mixture_tensor = self._normalize(self._pad_or_trim(mixture))
        vocals_tensor = self._normalize(self._pad_or_trim(vocals))

        # Logging normalized audio stats
        dataset_logger.debug(
            f"[After Padding/Normalizing] Mixture tensor - min: {mixture_tensor.min().item()}, "
            f"max: {mixture_tensor.max().item()}, mean: {mixture_tensor.mean().item()}, shape: {mixture_tensor.shape}"
        )
        dataset_logger.debug(
            f"[After Padding/Normalizing] Vocals tensor - min: {vocals_tensor.min().item()}, "
            f"max: {vocals_tensor.max().item()}, mean: {vocals_tensor.mean().item()}, shape: {vocals_tensor.shape}"
        )

        # Compute STFT directly in PyTorch
        window = torch.hann_window(self.n_fft, device=device, dtype=torch.bfloat16)
        mix_stft = torch.stft(
            mixture_tensor, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True
        )
        voc_stft = torch.stft(
            vocals_tensor, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True
        )

        # Compute magnitude
        mixture_mag_tensor = torch.abs(mix_stft).unsqueeze(0)
        vocals_mag_tensor = torch.abs(voc_stft).unsqueeze(0)

        # Logging STFT magnitude stats
        dataset_logger.debug(
            f"[STFT] Mixture Magnitude - min: {mixture_mag_tensor.min().item()}, "
            f"max: {mixture_mag_tensor.max().item()}, mean: {mixture_mag_tensor.mean().item()}, shape: {mixture_mag_tensor.shape}"
        )
        dataset_logger.debug(
            f"[STFT] Vocals Magnitude - min: {vocals_mag_tensor.min().item()}, "
            f"max: {vocals_mag_tensor.max().item()}, mean: {vocals_mag_tensor.mean().item()}, shape: {vocals_mag_tensor.shape}"
        )

        return mixture_mag_tensor, vocals_mag_tensor

    def _pad_or_trim(self, audio):
        max_length_samples = int(self.sr * self.max_length_seconds)
        audio_tensor = torch.tensor(audio, dtype=torch.bfloat16, device=device)
        if len(audio) < max_length_samples:
            padded_audio = F.pad(audio_tensor, (0, max_length_samples - len(audio_tensor)))
            dataset_logger.debug(
                f"[Pad] Padded audio to {max_length_samples} samples. Original length: {len(audio)}"
            )
            return padded_audio
        trimmed_audio = audio_tensor[:max_length_samples]
        dataset_logger.debug(
            f"[Trim] Trimmed audio to {max_length_samples} samples. Original length: {len(audio)}"
        )
        return trimmed_audio

    def _normalize(self, audio):
        audio_tensor = torch.tensor(audio, dtype=torch.bfloat16, device=device)
        max_val = torch.max(torch.abs(audio_tensor)) + 1e-8
        normalized_audio = audio_tensor / max_val
        dataset_logger.debug(
            f"[Normalize] Max value before normalization: {max_val.item():.5f}, "
            f"Min after normalization: {normalized_audio.min().item():.5f}, "
            f"Max after normalization: {normalized_audio.max().item():.5f}, "
            f"Mean after normalization: {normalized_audio.mean().item():.5f}"
        )
        return normalized_audio
