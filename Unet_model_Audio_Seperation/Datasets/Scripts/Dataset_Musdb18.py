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
from Datasets.Scripts.Dataset_utils import validate_audio, validate_spectrogram,_pad_or_trim,_normalize
root_dir = Return_root_dir()
train_log_path = os.path.join(root_dir, "Model_Performance_logg/log/datasets.txt")
data_logger = setup_logger('dataloader_logger', train_log_path)


device = torch.device("cpu")
class MUSDB18StemDataset(Dataset):
    def __init__(self, root_dir, subset='train', sr=44100, n_fft=1024, 
                 hop_length=512, max_length_seconds=5, max_files=100):
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



            #Checks if the mixture audio contains valid audio too be proccessed 
            if not validate_audio(mixture, self.sr, self.max_length_seconds):
                data_logger.warning(f"Invalid mixture audio in {file_path}")
                return None
            
            #Checks if the vocals audio contains valid audio too be proccessed 
            if not validate_audio(vocals, self.sr, self.max_length_seconds):
                data_logger.warning(f"Invalid vocals audio in {file_path}")
                return None 


    
            mixture_tensor = torch.from_numpy(mixture)
            vocals_tensor = torch.from_numpy(vocals)

            mixture_tensor = _normalize(_pad_or_trim(mixture_tensor, self.sr, self.max_length_seconds))
            vocals_tensor = _normalize(_pad_or_trim(vocals_tensor, self.sr, self.max_length_seconds))

    
            window = torch.hann_window(self.n_fft)

            mix_stft = torch.stft(mixture_tensor, n_fft=self.n_fft, hop_length=self.hop_length, 
                                  window=window, return_complex=True)
            
            voc_stft = torch.stft(vocals_tensor, n_fft=self.n_fft, hop_length=self.hop_length,
                                  window=window, return_complex=True)

          
            mixture_mag = torch.abs(mix_stft).unsqueeze(0)
            vocals_mag = torch.abs(voc_stft).unsqueeze(0)

            if not validate_spectrogram(mixture_mag):
                data_logger.warning(f"Invalid mixture spectrogram in {file_path}")
                return None
            
            if not validate_spectrogram(vocals_mag):
                data_logger.warning(f"Invalid vocals spectrogram in {file_path}")
                return None
            

            return mixture_mag, vocals_mag

        except Exception as e:
            data_logger.error(f"Error processing {os.path.basename(file_path)}: {str(e)}")
            return None



    def _to_mono(self, audio):
        return np.mean(audio, axis=1) if audio.ndim == 2 else audio


