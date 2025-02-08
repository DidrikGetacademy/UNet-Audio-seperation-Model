import os
import librosa
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import sys
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Training.Externals.utils import Return_root_dir
from Training.Externals.Logger import setup_logger

root_dir = Return_root_dir()
train_log_path = os.path.join(root_dir, "Model_Performance_logg/log/datasets.txt")
dataset_logger = setup_logger('dataset_DSD100', train_log_path)


device = torch.device("cpu")

class DSD100(Dataset):
    def __init__(self, root_dir, subset='Dev', sr=44100, n_fft=1024, 
                 hop_length=512, max_length_seconds=11, max_files=None):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length_seconds = max_length_seconds

        self.mixtures_dir = os.path.join(root_dir, 'Mixtures', subset)
        self.sources_dir = os.path.join(root_dir, 'Sources', subset)

        self.song_folders = sorted([
            song for song in os.listdir(self.mixtures_dir)
            if os.path.isdir(os.path.join(self.mixtures_dir, song))
        ])[:max_files]

        dataset_logger.info(f"Initialized DSD100 with {len(self.song_folders)} tracks")
        dataset_logger.info(f"STFT params: n_fft={n_fft}, hop={hop_length}")

    def __len__(self):
        return len(self.song_folders)

    def __getitem__(self, idx):
        song = self.song_folders[idx]
        mix_path = os.path.join(self.mixtures_dir, song, 'mixture.wav')
        voc_path = os.path.join(self.sources_dir, song, 'vocals.wav')

        try:
      
            mixture = self._load_audio(mix_path)
            vocals = self._load_audio(voc_path)

     
            mix_tensor = self._normalize(self._pad_or_trim(mixture))
            voc_tensor = self._normalize(self._pad_or_trim(vocals))

     
            window = torch.hann_window(self.n_fft, device=device)
            mix_stft = torch.stft(mix_tensor, n_fft=self.n_fft, hop_length=self.hop_length,
                                  window=window, return_complex=True)
            voc_stft = torch.stft(voc_tensor, n_fft=self.n_fft, hop_length=self.hop_length,
                                  window=window, return_complex=True)

        
            return torch.abs(mix_stft).unsqueeze(0), torch.abs(voc_stft).unsqueeze(0)

        except Exception as e:
            dataset_logger.error(f"Error processing {song}: {str(e)}")
            return None

    def _load_audio(self, path):
        audio, _ = librosa.load(path, sr=self.sr, mono=True)
        return audio.astype(np.float32)

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
