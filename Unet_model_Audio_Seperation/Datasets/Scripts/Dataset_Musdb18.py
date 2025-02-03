import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import stempeg
import librosa
import numpy as np
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
from Training.Externals.utils import Return_root_dir
from Training.Externals.Logger import setup_logger

# Setup logging
root_dir = Return_root_dir()
train_log_path = os.path.join(root_dir, "Model_Performance_logg/log/datasets.txt")
data_logger = setup_logger('dataloader_logger', train_log_path)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MUSDB18StemDataset(Dataset):
    def __init__(self, root_dir, subset='train', sr=44100, n_fft=1024, hop_length=512, max_length_seconds=15, max_files=None):
        self.root_dir = os.path.join(root_dir, subset)
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length_seconds = max_length_seconds

   
        self.file_paths = [
            os.path.join(self.root_dir, file)
            for file in os.listdir(self.root_dir)
            if file.endswith('.mp4')
        ]
        if max_files:
            self.file_paths = self.file_paths[:max_files]


        data_logger.info(f"Initialized MUSDB18 Dataset with {len(self.file_paths)} files in '{self.root_dir}' (subset='{subset}').")
        data_logger.info(f"Sample Rate: {self.sr}, N_FFT: {self.n_fft}, Hop Length: {self.hop_length}, Max Length: {self.max_length_seconds} sec")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            # Load stems (stems shape: [stem_index, samples, channels])
            stems, _ = stempeg.read_stems(file_path, sample_rate=self.sr)
            mixture, vocals = stems[0], stems[4]

            # Convert to mono
            mixture = self._to_mono(mixture)
            vocals = self._to_mono(vocals)

            # Random crop
            max_start_sample = len(mixture) - int(self.sr * self.max_length_seconds)
            start_sample = np.random.randint(0, max_start_sample)
            mixture = mixture[start_sample:start_sample + int(self.sr * self.max_length_seconds)]
            vocals = vocals[start_sample:start_sample + int(self.sr * self.max_length_seconds)]

            # Pad or trim
            mixture_tensor = self._normalize(self._pad_or_trim(mixture))
            vocals_tensor = self._normalize(self._pad_or_trim(vocals))

            # Logging raw audio info
            data_logger.debug(
                f"[Dataset] File: {os.path.basename(file_path)} | Mixture Shape: {mixture.shape}, "
                f"Min: {mixture.min():.5f}, Max: {mixture.max():.5f}, Mean: {mixture.mean():.5f} | "
                f"Vocals Shape: {vocals.shape}, Min: {vocals.min():.5f}, Max: {vocals.max():.5f}, Mean: {vocals.mean():.5f}"
            )

            # Compute STFT in PyTorch
            window = torch.hann_window(self.n_fft, device=device, dtype=torch.bfloat16)
            mix_stft = torch.stft(
                mixture_tensor, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True
            )
            voc_stft = torch.stft(
                vocals_tensor, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True
            )

            # Compute magnitude spectrograms
            mixture_mag_tensor = torch.abs(mix_stft).unsqueeze(0)  # (1, F, T)
            vocals_mag_tensor = torch.abs(voc_stft).unsqueeze(0)  # (1, F, T)

            # Log STFT magnitude details
            data_logger.debug(
                f"[STFT] Mixture Mag: Shape: {mixture_mag_tensor.shape}, Min: {mixture_mag_tensor.min().item():.5f}, "
                f"Max: {mixture_mag_tensor.max().item():.5f}, Mean: {mixture_mag_tensor.mean().item():.5f}"
            )
            data_logger.debug(
                f"[STFT] Vocals Mag: Shape: {vocals_mag_tensor.shape}, Min: {vocals_mag_tensor.min().item():.5f}, "
                f"Max: {vocals_mag_tensor.max().item():.5f}, Mean: {vocals_mag_tensor.mean().item():.5f}"
            )

            return mixture_mag_tensor, vocals_mag_tensor

        except Exception as e:
            data_logger.error(f"Error processing file {file_path}: {e}")
            return None

    def _to_mono(self, audio):
        if audio.ndim == 2:
            return np.mean(audio, axis=1)
        return audio

    def _pad_or_trim(self, audio):
        max_length_samples = int(self.sr * self.max_length_seconds)
        audio_tensor = torch.tensor(audio, dtype=torch.bfloat16, device=device)
        if len(audio) < max_length_samples:
            padded_audio = F.pad(audio_tensor, (0, max_length_samples - len(audio_tensor)))
            data_logger.debug(f"[Pad] Padded audio to {max_length_samples} samples (original: {len(audio)})")
            return padded_audio
        trimmed_audio = audio_tensor[:max_length_samples]
        data_logger.debug(f"[Trim] Trimmed audio to {max_length_samples} samples (original: {len(audio)})")
        return trimmed_audio

    def _normalize(self, audio):
        audio_tensor = torch.tensor(audio, dtype=torch.bfloat16, device=device)
        max_val = torch.max(torch.abs(audio_tensor)) + 1e-8
        normalized_audio = audio_tensor / max_val
        data_logger.debug(
            f"[Normalize] Max Before: {max_val.item():.5f} | Min After: {normalized_audio.min().item():.5f}, "
            f"Max After: {normalized_audio.max().item():.5f}, Mean: {normalized_audio.mean().item():.5f}"
        )
        return normalized_audio
