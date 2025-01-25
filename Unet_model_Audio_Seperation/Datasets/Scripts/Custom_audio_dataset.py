import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from Training.Externals.utils import Return_root_dir


from Training.Externals.Logger import setup_logger


root_dir = Return_root_dir() #Gets the root directory
train_log_path = os.path.join(root_dir, "Model_performance_logg/log/Model_Training_logg.txt")
data_logger = setup_logger( 'dataloader_logger',train_log_path)

class CustomAudioDataset(Dataset):
    def __init__(self, input_dir, target_dir, sr=44100, n_fft=1024, hop_length=512, max_length_seconds=10):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length_seconds = max_length_seconds
        self.input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.WAV')])
        self.target_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.WAV')])
        data_logger.info(f"Dataset initialized with inputfiles:  {len(self.input_files)} valid files. & targetfiles : {len(self.target_files)} valid files.")
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

        # Pad or trim to fixed length
        max_length_samples = int(self.sr * self.max_length_seconds)
        input_audio = self._pad_or_trim(input_audio, max_length_samples)
        target_audio = self._pad_or_trim(target_audio, max_length_samples)

        # Compute STFT
        input_stft = librosa.stft(input_audio, n_fft=self.n_fft, hop_length=self.hop_length)
        target_stft = librosa.stft(target_audio, n_fft=self.n_fft, hop_length=self.hop_length)

        # Get magnitude of STFT
        input_mag = np.abs(input_stft)
        target_mag = np.abs(target_stft)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Convert to tensors
        input_mag_tensor = torch.tensor(input_mag, dtype=torch.float32).unsqueeze(0).to(device)
        target_mag_tensor = torch.tensor(target_mag, dtype=torch.float32).unsqueeze(0).to(device)
        print(f"[Custom_dataset] Input_mag_tensor shape: {input_mag_tensor.shape}, "
                f"device: {input_mag_tensor.device}, dtype: {input_mag_tensor.dtype}, "
                f"target_mag_tensor shape: {target_mag_tensor.shape}, "
                f"device: {target_mag_tensor.device}, dtype: {target_mag_tensor.dtype}")

        return input_mag_tensor, target_mag_tensor

    def _pad_or_trim(self, audio, max_length_samples):
        if len(audio) < max_length_samples:
            return np.pad(audio, (0, max_length_samples - len(audio)), mode='constant')
        print(f"Returning audio with max_length_samples: {max_length_samples}")
        return audio[:max_length_samples]
