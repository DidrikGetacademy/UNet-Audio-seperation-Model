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
from Datasets.Scripts.Dataset_utils import validate_audio, validate_spectrogram,_pad_or_trim,_normalize

root_dir = Return_root_dir()
train_log_path = os.path.join(root_dir, "Model_Performance_logg/log/datasets.txt")
dataset_logger = setup_logger('dataset_DSD100', train_log_path)


class DSD100(Dataset):
    def __init__(self, root_dir, subset='Dev', sr=44100, n_fft=1024, 
                 hop_length=512, max_length_seconds=5, max_files=50):
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

           #Checks if the mixture audio contains valid audio too be proccessed 
            if not validate_audio(mixture, self.sr, self.max_length_seconds):
                dataset_logger.warning(f"Invalid mixture audio in {mix_path}")
                return None
            
            #Checks if the vocals audio contains valid audio too be proccessed 
            if not validate_audio(vocals, self.sr, self.max_length_seconds):
                dataset_logger.warning(f"Invalid vocals audio in {voc_path}")
                return None 
            

     
            mix_tensor = _normalize(_pad_or_trim(mixture, self.sr, self.max_length_seconds))
            voc_tensor = _normalize(_pad_or_trim(vocals, self.sr, self.max_length_seconds))

     
            window = torch.hann_window(self.n_fft)
            mix_stft = torch.stft(mix_tensor, n_fft=self.n_fft, hop_length=self.hop_length,
                                  window=window, return_complex=True)
            voc_stft = torch.stft(voc_tensor, n_fft=self.n_fft, hop_length=self.hop_length,
                                  window=window, return_complex=True)

            mixture_mag = torch.abs(mix_stft).unsqueeze(0)
            vocals_mag = torch.abs(voc_stft).unsqueeze(0)

            if not validate_spectrogram(mixture_mag):
                dataset_logger.warning(f"Invalid mixture spectrogram in {mix_path}")
                return None
            
            if not validate_spectrogram(vocals_mag):
                dataset_logger.warning(f"Invalid vocals spectrogram in {voc_path}")
                return None
            
            return  mixture_mag, vocals_mag

        except Exception as e:
            dataset_logger.error(f"Error processing {song}: {str(e)}")
            return None


    def _load_audio(self, path):
        audio, _ = librosa.load(path, sr=self.sr, mono=True)
        return audio.astype(np.float32)



 