import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import stempeg
from Training.Externals.Logger import setup_logger

# Setup logger
data_logger = setup_logger( 'dataloader_logger', r'C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_performance_logg\log\Model_Training_logg.txt')


class MUSDB18StemDataset(Dataset):
    def __init__(self, root_dir, subset='train', sr=44100, n_fft=2048, hop_length=512, max_length_seconds=5, max_files=None, validate_files=False):
        self.root_dir = os.path.join(root_dir, subset)
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length_seconds = max_length_seconds

        # Load file paths
        self.file_paths = [
            os.path.join(self.root_dir, file)
            for file in os.listdir(self.root_dir)
            if file.endswith('.mp4')
        ]

        if max_files:
            self.file_paths = self.file_paths[:max_files]

        data_logger.info(f"Found {len(self.file_paths)} files in '{self.root_dir}' (subset='{subset}')")

        if validate_files:
            self.file_paths = self._validate_files(self.file_paths)
        data_logger.info(f"Dataset initialized with {len(self.file_paths)} valid files.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            stems, _ = stempeg.read_stems(file_path, sample_rate=self.sr)
            mixture, vocals = stems[0], stems[4]

            # Process mixture and vocals
            mixture = self._normalize(self._pad_or_trim(self._to_mono(mixture)))
            vocals = self._normalize(self._pad_or_trim(self._to_mono(vocals)))

            # Compute STFTs
            mixture_stft = librosa.stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length)
            vocals_stft = librosa.stft(vocals, n_fft=self.n_fft, hop_length=self.hop_length)

            # Magnitude
            mixture_mag = self._adjust_length(np.abs(mixture_stft))
            vocals_mag = self._adjust_length(np.abs(vocals_stft))

            return (
                torch.tensor(mixture_mag, dtype=torch.float32).unsqueeze(0),
                torch.tensor(vocals_mag, dtype=torch.float32).unsqueeze(0),
            )
        except Exception as e:
            data_logger.error(f"Error processing file {file_path}: {e}")
            return None

    def _to_mono(self, audio):
        return np.mean(audio, axis=1) if audio.ndim == 2 else audio

    def _pad_or_trim(self, audio):
        max_length_samples = int(self.sr * self.max_length_seconds)
        if len(audio) < max_length_samples:
            return np.pad(audio, (0, max_length_samples - len(audio)), mode='constant')
        return audio[:max_length_samples]

    def _normalize(self, audio):
        return audio / (np.max(np.abs(audio)) + 1e-8)

    def _adjust_length(self, spectrogram):
        desired_time_dim = (self.max_length_seconds * self.sr - self.n_fft) // self.hop_length + 1
        time_dim = spectrogram.shape[1]
        if time_dim < desired_time_dim:
            return np.pad(spectrogram, ((0, 0), (0, desired_time_dim - time_dim)), mode='constant')
        return spectrogram[:, :desired_time_dim]

    def _validate_files(self, file_paths):
        valid_files = []
        for file in file_paths:
            try:
                audio, _ = librosa.load(file, sr=self.sr, mono=True)
                if self._has_valid_audio(audio):
                    valid_files.append(file)
                else:
                    data_logger.warning(f"Skipping invalid audio file: {file}")
            except Exception as e:
                data_logger.error(f"Error loading file '{file}': {e}")
        return valid_files

    def _has_valid_audio(self, audio, threshold=0.01):
        return np.max(np.abs(audio)) > threshold