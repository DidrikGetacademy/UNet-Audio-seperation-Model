import os
import librosa
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)

from Training.Externals.utils import Return_root_dir
from Training.Externals.Logger import setup_logger
from Datasets.Scripts.Dataset_utils import validate_audio, validate_spectrogram, _pad_or_trim, _normalize

root_dir = Return_root_dir()
train_log_path = os.path.join(root_dir, "Model_Performance_logg/log/datasets.txt")
data_logger = setup_logger('custom_dataset', train_log_path)

class CustomAudioDataset(Dataset):
    def __init__(self, root_dir, sr=44100, n_fft=1024,
                 hop_length=512, max_length_seconds=None, max_files=10):
        self.input_dir = os.path.join(root_dir, "input")   
        self.target_dir = os.path.join(root_dir, "target")  
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length_seconds = max_length_seconds

        # Sorter og begrens filer
        self.input_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.wav')])[:max_files]
        self.target_files = sorted([f for f in os.listdir(self.target_dir) if f.endswith('.wav')])[:max_files]

        # Valider filpar
        if len(self.input_files) != len(self.target_files):
            raise ValueError("Mismatch mellom antall input- og target-filer")

        data_logger.info(f"Initialized CustomDataset med {len(self.input_files)} filpar")
        data_logger.info(f"STFT-konfig: n_fft={n_fft}, hop={hop_length}")

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        try:
            # Laster rålyd
            input_waveform = self._load_waveform(os.path.join(self.input_dir, self.input_files[idx]))
            target_waveform = self._load_waveform(os.path.join(self.target_dir, self.target_files[idx]))

            # Trimm/pad med samme start_index
            max_samples = int(self.sr * self.max_length_seconds)
            input_waveform, target_waveform, start_index = self._synchronized_pad_trim(
                input_waveform, 
                target_waveform, 
                max_samples
            )

            # Valider amplituden
            if not self._validate_amplitude(input_waveform, target_waveform):
                data_logger.warning(f"low amplitude in {self.input_files[idx]}")
                return None

            # Normaliser for spektrogram
            input_normalized = _normalize(input_waveform.clone())
            target_normalized = _normalize(target_waveform.clone())

            # Beregn spektrogrammer
            window = torch.hann_window(self.n_fft)
            input_stft = torch.stft(
                input_normalized, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length,
                window=window, 
                return_complex=True, 
                center=False
            )
            target_stft = torch.stft(
                target_normalized,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                return_complex=True,
                center=False
            )

            # Magnituder
            mixture_mag = torch.abs(input_stft).unsqueeze(0)  # Input er mixture
            vocal_mag = torch.abs(target_stft).unsqueeze(0)   # Target er vokal

            # Valider spektrogrammer
            if not validate_spectrogram(mixture_mag) or not validate_spectrogram(vocal_mag):
                data_logger.warning(f"Ugyldig spektrogram i {self.input_files[idx]}")
                return None

            return mixture_mag, vocal_mag, target_waveform  # Returner rå vokal-lyd

        except Exception as e:
            data_logger.error(f"Feil under prosessering av filpar {idx}: {str(e)}")
            return None

    def _load_waveform(self, path):
        """Laster rå lyd uten normalisering."""
        audio, _ = librosa.load(path, sr=self.sr, mono=True)
        return torch.from_numpy(audio.astype(np.float32))

    def _synchronized_pad_trim(self, input_audio, target_audio, max_samples):
        """Trimmer/padder begge lydene med samme start_index."""
        start_index = None
        max_attempts = 15

        for _ in range(max_attempts):
            max_start = min(len(input_audio), len(target_audio)) - max_samples
            if max_start <= 0:
                start_index = 0
                break

            start_index = torch.randint(0, max_start + 1, ()).item()
            
            input_snippet = input_audio[start_index : start_index + max_samples]
            target_snippet = target_audio[start_index : start_index + max_samples]

            if (np.max(np.abs(input_snippet)) > 0.05 and 
                np.max(np.abs(target_snippet)) > 0.05):
                break
        else:
            raise ValueError("Kunne ikke finne segment med tilstrekkelig amplitude")

        # Anvend start_index på begge lydene
        input_padded = _pad_or_trim(input_audio, self.sr, self.max_length_seconds, start_index)
        target_padded = _pad_or_trim(target_audio, self.sr, self.max_length_seconds, start_index)

        return input_padded, target_padded, start_index

    def _validate_amplitude(self, input_audio, target_audio):
        input_energy = torch.mean(input_audio ** 2)
        target_energy = torch.mean(target_audio ** 2)
        return (input_energy > 1e-6) and (target_energy > 1e-6)