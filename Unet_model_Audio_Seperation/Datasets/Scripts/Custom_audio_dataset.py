import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from Training.Externals.utils import Return_root_dir
from Training.Externals.Logger import setup_logger

root_dir = Return_root_dir()  # Gets the root directory
train_log_path = os.path.join(root_dir, "Model_Performance_logg/log/datasets.txt")
data_logger = setup_logger('dataloader_logger', train_log_path)
class CustomAudioDataset(Dataset):
    def __init__(
        self, root_dir, 
        sr=44100, 
        n_fft=1024,
        hop_length=512,
        max_length_seconds=15,
        max_files=None
        ):
        
        self.input_dir = os.path.join(root_dir, "input")
        self.target_dir = os.path.join(root_dir, "target")
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length_seconds = max_length_seconds

        # Hent filnavn fra input og target-mapper
        self.input_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.WAV')])
        self.target_files = sorted([f for f in os.listdir(self.target_dir) if f.endswith('.WAV')])

        data_logger.info(f"Dataset initialized with {len(self.input_files)} input files & {len(self.target_files)} target files.")
        assert len(self.input_files) == len(self.target_files), "Mismatch between input and target files."

        if max_files is not None:
            self.input_files = self.input_files[:max_files]
            self.target_files = self.target_files[:max_files]

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        target_path = os.path.join(self.target_dir, self.target_files[idx])

        input_audio, _ = librosa.load(input_path, sr=self.sr)
        target_audio, _ = librosa.load(target_path, sr=self.sr)

        # Normalisering
        input_audio = input_audio / (abs(input_audio).max() + 1e-8)
        target_audio = target_audio / (abs(target_audio).max() + 1e-8)

        # Padding / trimming
        max_length_samples = int(self.sr * self.max_length_seconds)
        input_audio = self._pad_or_trim(input_audio, max_length_samples)
        target_audio = self._pad_or_trim(target_audio, max_length_samples)

        # STFT
        input_stft = librosa.stft(input_audio, n_fft=self.n_fft, hop_length=self.hop_length)
        target_stft = librosa.stft(target_audio, n_fft=self.n_fft, hop_length=self.hop_length)

        # STFT Magnitude
        input_mag = np.abs(input_stft)
        target_mag = np.abs(target_stft)

        # Konverter til PyTorch-tensorer
        input_mag_tensor = torch.tensor(input_mag, dtype=torch.float32).unsqueeze(0)
        target_mag_tensor = torch.tensor(target_mag, dtype=torch.float32).unsqueeze(0)

        return input_mag_tensor, target_mag_tensor

    def _pad_or_trim(self, audio, max_length_samples):
        if len(audio) < max_length_samples:
            return np.pad(audio, (0, max_length_samples - len(audio)), mode='constant')
        return audio[:max_length_samples]
