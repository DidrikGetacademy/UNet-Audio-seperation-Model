import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import stempeg
import numpy as np
import sys


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Training.Externals.utils import Return_root_dir
from Training.Externals.Logger import setup_logger

root_dir = Return_root_dir()
train_log_path = os.path.join(root_dir, "Model_Performance_logg/log/datasets.txt")
data_logger = setup_logger('dataloader_logger', train_log_path)


device = torch.device("cpu")

class MUSDB18StemDataset(Dataset):
    def __init__(self, root_dir, subset='train', sr=44100, n_fft=1024, 
                 hop_length=512, max_length_seconds=15, max_files=None):
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

        data_logger.info(f"Initialized MUSDB18 Dataset with {len(self.file_paths)} files")
        data_logger.info(f"Config: SR={sr}, N_FFT={n_fft}, Hop={hop_length}, MaxLen={max_length_seconds}s")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
      
            stems, _ = stempeg.read_stems(file_path, sample_rate=self.sr)
            mixture, vocals = stems[0], stems[4]

            mixture = self._to_mono(mixture).astype(np.float32)
            vocals = self._to_mono(vocals).astype(np.float32)

    
            max_length_samples = int(self.sr * self.max_length_seconds)
            max_start = len(mixture) - max_length_samples
            start = np.random.randint(0, max_start) if max_start > 0 else 0
            end = start + max_length_samples
            mixture = mixture[start:end]
            vocals = vocals[start:end]

    
            mixture_tensor = self._normalize(self._pad_or_trim(mixture))
            vocals_tensor = self._normalize(self._pad_or_trim(vocals))

    
            window = torch.hann_window(self.n_fft, device=device)
            mix_stft = torch.stft(mixture_tensor, n_fft=self.n_fft, hop_length=self.hop_length, 
                                  window=window, return_complex=True)
            voc_stft = torch.stft(vocals_tensor, n_fft=self.n_fft, hop_length=self.hop_length,
                                  window=window, return_complex=True)

          
            mixture_mag = torch.abs(mix_stft).unsqueeze(0)
            vocals_mag = torch.abs(voc_stft).unsqueeze(0)
            return mixture_mag, vocals_mag

        except Exception as e:
            data_logger.error(f"Error processing {os.path.basename(file_path)}: {str(e)}")
            return None

    def _to_mono(self, audio):
        return np.mean(audio, axis=1) if audio.ndim == 2 else audio

    def _pad_or_trim(self, audio):
        max_samples = int(self.sr * self.max_length_seconds)
        audio_tensor = torch.from_numpy(audio).to(device)
        if len(audio) < max_samples:
            return F.pad(audio_tensor, (0, max_samples - len(audio)))
        return audio_tensor[:max_samples]

    def _normalize(self, audio):
        audio = audio.clone().detach()
        max_val = torch.max(torch.abs(audio)) + 1e-8
        return audio / max_val

