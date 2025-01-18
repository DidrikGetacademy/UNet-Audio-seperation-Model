import os
import torch
from torch.utils.data import Dataset
import librosa

class CustomAudioDataset(Dataset):
    def __init__(self, input_dir, target_dir, sr=44100):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.sr = sr
        self.input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.WAV')])
        self.target_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.WAV')])
        assert len(self.input_files) == len(self.target_files), "Mismatch between input and target files."

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        target_path = os.path.join(self.target_dir, self.target_files[idx])

        input_audio, _ = librosa.load(input_path, sr=self.sr)
        target_audio, _ = librosa.load(target_path, sr=self.sr)

        # Normalize audio
        input_audio = input_audio / (abs(input_audio).max() + 1e-8)
        target_audio = target_audio / (abs(target_audio).max() + 1e-8)

        # Convert to torch tensors
        return torch.tensor(input_audio, dtype=torch.float32).unsqueeze(0), torch.tensor(target_audio, dtype=torch.float32).unsqueeze(0)
