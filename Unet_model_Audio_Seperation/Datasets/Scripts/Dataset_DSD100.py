# dataset1.py
import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Training.Externals.Logger import setup_logger
from Training.Externals.utils import Return_root_dir



root_dir = Return_root_dir() #Gets the root directory
train_log_path = os.path.join(root_dir, "Model_performance_logg/log/Model_Training_logg.txt")
dataset_logger = setup_logger('dataset_Musdb18',train_log_path)



    #DSD100 dataset returning (mixture_mag, vocals_mag).
    #Each is shape [1, freq, time].
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

        # Song folders under Mixtures
        self.song_folders = [
            song for song in os.listdir(self.mixtures_dir)
            if os.path.isdir(os.path.join(self.mixtures_dir, song))
        ]
        if max_files is not None:
            self.song_folders = self.song_folders[:max_files]

    def __len__(self):
        return len(self.song_folders)

    def __getitem__(self, idx):
        song_folder = self.song_folders[idx]

        #Build full paths
        mixture_path = os.path.join(self.mixtures_dir, song_folder, 'mixture.wav')
        vocals_path = os.path.join(self.sources_dir, song_folder, 'vocals.wav')

        #Load audio as mono
        mixture, _ = librosa.load(mixture_path, sr=self.sr, mono=True)
        vocals, _ = librosa.load(vocals_path, sr=self.sr, mono=True)

        #Pad or trim to exact # of samples
        mixture = self._normalize(self._pad_or_trim(mixture))
        vocals = self._normalize(self._pad_or_trim(vocals))

        #STFT
        mix_stft = librosa.stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length)
        voc_stft = librosa.stft(vocals, n_fft=self.n_fft, hop_length=self.hop_length)

        #Magnitude
        mixture_mag = np.abs(mix_stft)
        vocals_mag = np.abs(voc_stft)

        #Adjust shape so time dimension matches the formula
        mixture_mag = self._adjust_length(mixture_mag)
        vocals_mag = self._adjust_length(vocals_mag)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DSD100] Moving tensors to {device}...")
        #Return in shape [1, freq, time]
        mixture_mag_tensor = torch.tensor(mixture_mag, dtype=torch.float32).unsqueeze(0).to(device)
        vocals_mag_tensor = torch.tensor(vocals_mag, dtype=torch.float32).unsqueeze(0).to(device)
        print(f"[DSD100] Tensors moved to {device}.")
        
        return mixture_mag_tensor, vocals_mag_tensor
        
    def _pad_or_trim(self, audio):
        # 10 s => 441000 samples if sr=44100
        max_length_samples = int(self.sr * self.max_length_seconds)
        if len(audio) < max_length_samples:
            return np.pad(audio, (0, max_length_samples - len(audio)), mode='constant')
        return audio[:max_length_samples]

    def _normalize(self, audio):
        return audio / (np.max(np.abs(audio)) + 1e-8)

    def _adjust_length(self, spectrogram):
        desired_time_dim = (self.max_length_seconds * self.sr - self.n_fft) // self.hop_length + 1
        current_time_dim = spectrogram.shape[1]
        if current_time_dim < desired_time_dim:
            # pad right
            return np.pad(
                spectrogram,
                ((0, 0), (0, desired_time_dim - current_time_dim)),
                mode='constant'
            )
        else:
            # crop if it's bigger
            return spectrogram[:, :desired_time_dim]
