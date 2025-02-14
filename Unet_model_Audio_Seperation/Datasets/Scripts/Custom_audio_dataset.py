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
from Datasets.Scripts.Dataset_utils import validate_audio, validate_spectrogram
root_dir = Return_root_dir()
train_log_path = os.path.join(root_dir, "Model_Performance_logg/log/datasets.txt")
data_logger = setup_logger('custom_dataset', train_log_path)
from Datasets.Scripts.Dataset_utils import validate_audio, validate_spectrogram,_pad_or_trim,_normalize

class CustomAudioDataset(Dataset):
    def __init__(self, root_dir, sr=44100, n_fft=1024,
                 hop_length=512, max_length_seconds=None, max_files=None):
        self.input_dir = os.path.join(root_dir, "input")
        self.target_dir = os.path.join(root_dir, "target")
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length_seconds = max_length_seconds

        self.input_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.wav')])[:max_files]
        self.target_files = sorted([f for f in os.listdir(self.target_dir) if f.endswith('.wav')])[:max_files]

        data_logger.info(f"Initialized CustomDataset with {len(self.input_files)} pairs")
        data_logger.info(f"STFT Config: n_fft={n_fft}, hop={hop_length}")

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        try:

            input_audio = self._process_audio(os.path.join(self.input_dir, self.input_files[idx]))
            target_audio = self._process_audio(os.path.join(self.target_dir, self.target_files[idx]))


            target_audio =_pad_or_trim(target_audio, self.max_length_seconds)
            input_audio =_pad_or_trim(input_audio, self.max_length_seconds)

            input_audio = _normalize(input_audio)
            target_audio = _normalize(target_audio)
            
            if not validate_audio(input_audio, self.sr, self.max_length_seconds):
                data_logger.warning(f"Invalid mixture audio in {self.input_files}")
                return None
            if not validate_audio(target_audio, self.sr, self.max_length_seconds):
                data_logger.warning(f"Invalid vocals audio in {self.target_files}")
                return None 
            

            window = torch.hann_window(self.n_fft)
            input_stft = torch.stft(input_audio, n_fft=self.n_fft, hop_length=self.hop_length,
                                    window=window, return_complex=True)
            target_stft = torch.stft(target_audio, n_fft=self.n_fft, hop_length=self.hop_length,
                                     window=window, return_complex=True)
            
            vocal_mag = torch.abs(input_stft).unsqueeze(0)
            mixture_mag = torch.abs(target_stft).unsqueeze(0)


            if not validate_spectrogram(vocal_mag) or not validate_spectrogram(mixture_mag):
                print("Invalid spectrogram")
                return None
            
            return vocal_mag, mixture_mag

        except Exception as e:
            data_logger.error(f"Error processing file pair {idx}: {str(e)}")
            return None


    def _process_audio(self, path):

        audio, _ = librosa.load(path, sr=self.sr, mono=True)
        audio = audio.astype(np.float32)

        audio_tensor = torch.from_numpy(audio).to(torch.device("cpu"))
        
        max_samples = int(self.sr * self.max_length_seconds)
        if len(audio) < max_samples:
            audio_tensor = F.pad(audio_tensor, (0, max_samples - len(audio)))
        else:
            audio_tensor = audio_tensor[:max_samples]
        # Normaliser
        max_val = torch.max(torch.abs(audio_tensor)) + 1e-8
        return audio_tensor / max_val
