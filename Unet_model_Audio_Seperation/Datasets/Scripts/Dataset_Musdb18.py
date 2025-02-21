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
from Datasets.Scripts.Dataset_utils import validate_audio,validate_alignment, validate_spectrogram,_pad_or_trim,_normalize
root_dir = Return_root_dir()
train_log_path = os.path.join(root_dir, "Model_Performance_logg/log/datasets.txt")
data_logger = setup_logger('dataloader_logger', train_log_path)


device = torch.device("cpu")
class MUSDB18StemDataset(Dataset):
    def __init__(self, root_dir, subset='train', sr=44100, n_fft=1024, 
                 hop_length=512, max_length_seconds=10, max_files=100):
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
            if stems.shape[0] < 5:
                data_logger.warning(f"[MUSDB18]--->Not enough stems in [musdb18] {file_path}")
                return None

            mixture, vocals = stems[0], stems[4]


            mixture = self._to_mono(mixture).astype(np.float32)
            vocals = self._to_mono(vocals).astype(np.float32)
            

            max_samples = int(self.sr * self.max_length_seconds)
            start_index = None
            if mixture.shape[0] > max_samples and vocals.shape[0] > max_samples:
                max_start = min(mixture.shape[0] - max_samples, vocals.shape[0] - max_samples)
                start_index = torch.randint(0, max_start + 1,()).item()


            mixture_tensor = _pad_or_trim(mixture, self.sr, self.max_length_seconds, start_index)
            vocals_tensor =  _pad_or_trim(vocals, self.sr, self.max_length_seconds, start_index)


            expected_length = int(self.sr * self.max_length_seconds)

            if len(mixture_tensor) != expected_length:
                data_logger.warning(f"[MUSDB18]--->Mixture length does not match expected length! (Expected: {expected_length}, Got: {len(mixture_tensor)}), Trimming now")
                mixture_tensor = mixture_tensor[:expected_length]
                
            
            if len(vocals_tensor) != expected_length:
                data_logger.warning(f"[MUSDB18]--->Vocals length does not match expected length! (Expected: {expected_length}, Got: {len(vocals_tensor)}, Trimming now..")
                vocals_tensor = vocals_tensor[:expected_length]

        
            mixture_np = mixture_tensor.numpy()
            vocals_np = vocals_tensor.numpy()


            if not validate_audio(mixture_np, self.sr, self.max_length_seconds):
                data_logger.warning(f"[MUSDB18]--->Invalid mixture audio in {file_path}")
                return None

            if not validate_audio(vocals_np, self.sr, self.max_length_seconds):
                data_logger.warning(f"[MUSDB18]--->Invalid vocals audio in {file_path}")
                return None


            mixture_tensor = _normalize(mixture_tensor)
            vocals_tensor = _normalize(vocals_tensor)

            if not validate_alignment(mixture_tensor, vocals_tensor, self.sr):
                data_logger.warning(f"[MUSDB18] ---> Alignment invalid for file {file_path}")
                return None

    
            window = torch.hann_window(self.n_fft)
            mix_stft = torch.stft(mixture_tensor, n_fft=self.n_fft, hop_length=self.hop_length, 
                                  window=window, return_complex=True,center=False)
            
            voc_stft = torch.stft(vocals_tensor, n_fft=self.n_fft, hop_length=self.hop_length,
                                  window=window, return_complex=True,center=False)

          
            mixture_mag = torch.abs(mix_stft).unsqueeze(0)
            vocals_mag = torch.abs(voc_stft).unsqueeze(0)

            if not validate_spectrogram(mixture_mag):
                data_logger.warning(f"[MUSDB18]--->Invalid mixture spectrogram in {file_path}")
                return None
            
            if not validate_spectrogram(vocals_mag):
                data_logger.warning(f"[MUSDB18]--->Invalid vocals spectrogram in {file_path}")
                return None
            

            return mixture_mag, vocals_mag

        except Exception as e:
            data_logger.error(f"[MUSDB18]---> Error processing {os.path.basename(file_path)}: {str(e)}")
            return None



    def _to_mono(self, audio):
        return np.mean(audio, axis=1) if audio.ndim == 2 else audio


