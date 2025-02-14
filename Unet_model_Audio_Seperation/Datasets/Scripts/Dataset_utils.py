import os
import sys
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Training.Externals.utils import Return_root_dir
from Training.Externals.Logger import setup_logger

root_dir = Return_root_dir()
train_log_path = os.path.join(root_dir, "Model_Performance_logg/log/datasets.txt")
data_logger = setup_logger('dataset_DSD100', train_log_path)


def _pad_or_trim(audio, sr, max_length_seconds):
    max_samples = int(sr * max_length_seconds)

    if isinstance(audio,np.ndarray):
         audio = torch.from_numpy(audio)

    audio_len = audio.shape[0]
        
    if audio_len > max_samples:
            max_start = audio_len - max_samples
            start = torch.randint(0, max_start + 1,()).item() if max_start > 0 else 0
            audio = audio[start : start + max_samples]
    else:
            pad_amount = max_samples - audio_len
            audio = F.pad(audio.unsqueeze(0), (0, pad_amount), mode='constant',value=0).squeeze(0)
    return audio



def _normalize(audio):
        max_val = torch.max(torch.abs(audio)) + 1e-8

        if max_val < 0.05:
           data_logger.info(f"Normalizing: amplitude very low ({max_val:.3f}), consider discarding or amplifying signal.")
        return audio / max_val



def validate_audio(audio, sr, max_length_seconds, min_amplitude=0.05):
        expected_length = int(sr * max_length_seconds)
        data_logger.info(f"Expected_length [validate_audio]: {expected_length}")

        if len(audio) == 0:
            data_logger.info(f"Empty Audio file")
            return False
        
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
           data_logger.info(f"Invalid values in audio file [(NaNs or Infs)]")
           return False
        
        max_amp = np.max(np.abs(audio))


        if max_amp > 1:
             data_logger.info("Unexpected amplitude range (amplitude > 1)")
             return False
        
        if max_amp < min_amplitude:
             data_logger.info("Unexpected amplitude range (amplitude < 0.05), Audio amplitude too low.")
             return False


        
        if np.max(np.abs(audio)) > 1:
            data_logger.info(f"unexpected amplitude range...")
            return False
        
        if len(audio) < expected_length * 0.5:
            data_logger.warning(f"Audio is too short: {len(audio)} samples  (expected {expected_length})")
            return False
        
        if len(audio) > expected_length:
            data_logger.warning(f"Audio is too long: {len(audio)}")
        return True
    


def validate_spectrogram(spectrogram):
        if spectrogram is None or spectrogram.size(0) == 0:
            data_logger.info(f"Spectrogram is None")
            return False
        
        if torch.any(torch.isnan(spectrogram)) or torch.any(torch.isinf(spectrogram)):
            data_logger.info(f"Spectrogram is nan of inf")
            return False
        
        if spectrogram.shape[0] == 0 or spectrogram.shape[1] == 0:
            data_logger.info(f"Spectro gram shape is invalid/0")
            return False
        
        return True




def spectrogram_to_waveform(magnitude_spectrogram, n_fft=1024, hop_length=512, num_iters=32, win_length=None):
    magnitude_spectrogram = magnitude_spectrogram.squeeze(0).to(torch.float32)
    window = torch.hann_window(win_length or n_fft, dtype=torch.float32)  
    waveform = torchaudio.functional.griffinlim(
        magnitude_spectrogram, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length or n_fft, 
        window=window, 
        power=1, 
        n_iter=num_iters,
        momentum=0.99, 
        length=None, 
        rand_init=True
    )
    return waveform


def Convert_spectrogram_to_audio(audio_path,predicted_vocals=None,targets=None,inputs=None,outputs=None):
        if inputs != None:
          input_waveform = spectrogram_to_waveform(inputs.clone().detach().to("cpu"))
          torchaudio.save(os.path.join( audio_path,"inputs/inputs(1).wav"), input_waveform.unsqueeze(0), sample_rate=44100)

        if predicted_vocals != None:
             predicted_waveform = spectrogram_to_waveform(predicted_vocals.clone().detach().to("cpu"))
             torchaudio.save(os.path.join(audio_path,"predictions/predictions(1).wav"), predicted_waveform.unsqueeze(0), sample_rate=44100)

        if targets != None:
          target_waveform = spectrogram_to_waveform(targets.clone().detach().to("cpu"))
          torchaudio.save(os.path.join(audio_path,"targets/targets(1).wav"), target_waveform.unsqueeze(0), sample_rate=44100)

        if outputs != None:
           output_waveform = spectrogram_to_waveform(outputs.clone().detach().to("cpu"))
           torchaudio.save(os.path.join(audio_path,"outputs/outputs(1).wav"), output_waveform.unsqueeze(0), sample_rate=44100)


