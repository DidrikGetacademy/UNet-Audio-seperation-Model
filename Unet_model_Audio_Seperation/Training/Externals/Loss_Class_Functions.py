import torch
import os
import sys
import torch.nn as nn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Training.Externals.Logger import setup_logger
from Training.Externals.utils import Return_root_dir

root_dir = Return_root_dir()  # Gets the root directory
train_log_path = os.path.join(root_dir, "Model_performance_logg/log/Model_Training_logg.txt")

train_logger = setup_logger('train', train_log_path)


# MASK-ESTIMATION LOSS CLASS
class MaskEstimationLoss(nn.Module):
    def __init__(self):
        super(MaskEstimationLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted_mask, mixture, target):
        # Move calculations to FP32 for stability
        predicted_mask = predicted_mask.float()
        mixture = mixture.float()
        target = target.float()

        # Estimated vocal spectrum from mask
        predicted_vocals = predicted_mask * mixture

        # L1 Loss
        l1 = self.l1_loss(predicted_vocals, target)

        # Log-space MSE Loss
        stft = self.mse_loss(torch.log1p(predicted_vocals), torch.log1p(target))

        # Weighted combination of losses
        return 0.5 * l1 + 0.5 * stft


# HYBRID-LOSS CLASS
class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        # Move calculations to FP32 for stability
        pred = pred.float()
        target = target.float()

        # L1 Loss
        l1 = self.l1_loss(pred, target)

        # STFT Loss
        n_fft = min(1024, pred.size(-1))
        hop_length = 512
        window = torch.hann_window(n_fft, device=pred.device, dtype=torch.float32)

        stft_loss = 0.0
        for i in range(pred.size(0)):  # Iterate over the batch
            pred_tensor = pred[i].squeeze()
            target_tensor = target[i].squeeze()

            # Ensure tensors have the same length
            min_len = min(pred_tensor.size(-1), target_tensor.size(-1))
            pred_tensor = pred_tensor[..., :min_len]
            target_tensor = target_tensor[..., :min_len]

            # Compute STFT
            pred_stft = torch.stft(pred_tensor, n_fft=n_fft, hop_length=hop_length, return_complex=True, window=window)
            target_stft = torch.stft(target_tensor, n_fft=n_fft, hop_length=hop_length, return_complex=True, window=window)

            # Log-space MSE Loss
            pred_stft_log = torch.log1p(torch.abs(pred_stft))
            target_stft_log = torch.log1p(torch.abs(target_stft))
            stft_loss_batch = self.mse_loss(pred_stft_log, target_stft_log)
            stft_loss += stft_loss_batch

            # Debugging
            train_logger.debug(f"[HybridLoss] Batch {i}: STFT MSE={stft_loss_batch.item():.6f}")

        stft_loss /= pred.size(0)  # Normalize by batch size

        # Combine losses
        combined_loss = 0.5 * l1 + 0.5 * stft_loss
        train_logger.debug(f"[HybridLoss] L1={l1.item():.6f}, STFT={stft_loss.item():.6f}, Combined={combined_loss.item():.6f}")

        return combined_loss, l1, stft_loss


# COMBINED LOSS CLASS
class Combinedloss(nn.Module):
    def __init__(self):
        super(Combinedloss, self).__init__()
        self.mask_loss = MaskEstimationLoss()
        self.hybrid_loss = HybridLoss()

    def forward(self, predicted_mask, mixture, target):
  
        predicted_mask = predicted_mask.float()
        mixture = mixture.float()
        target = target.float()

        # Mask Loss
        mask_loss = self.mask_loss(predicted_mask, mixture, target)

        # Hybrid Loss
        predicted_vocals = predicted_mask * mixture
        hybrid_loss_val, l1_loss_val, stft_loss_val  = self.hybrid_loss(predicted_vocals, target)

        # Combine losses
        combined_loss = 0.5 * mask_loss + 0.7 * hybrid_loss_val

        # Debugging
        train_logger.debug(f"[CombinedLoss] Mask Loss={mask_loss.item():.6f}, Hybrid Loss={hybrid_loss_val.item():.6f}")
        return combined_loss, mask_loss, hybrid_loss_val, l1_loss_val, stft_loss_val












####Fine Tuning####
class Combinedloss_Fine_tuning(nn.Module):
    def __init__(self,Fine_tune_logger, device):
        super(Combinedloss_Fine_tuning, self).__init__()
        self.device = device
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        Fine_tune_logger.info("Loading VGGish model for perceptual loss...")
        self.audio_model = torch.hub.load('harritaylor/torchvggish', 'vggish', trust_repo=True).to(device)
        self.audio_model.eval()
        Fine_tune_logger.info("VGGish model loaded and set to evaluation mode.")





    def preprocess_for_vggish(self, waveforms, fs=16000):
        waveforms = torch.stack([torch.from_numpy(w).float() for w in waveforms], dim=0).to(self.device)
 
        waveforms = waveforms / (waveforms.abs().max(dim=1, keepdim=True)[0] + 1e-9)
        return waveforms





    def spectrogram_to_waveform(self,Fine_tune_logger, spectrogram, n_fft=1024, hop_length=512):
        import librosa
        Fine_tune_logger.debug("Converting spectrogram(s) to waveform(s) using Griffin-Lim.")
        spectrogram_np = spectrogram.detach().cpu().numpy()
  
        if spectrogram_np.ndim == 3:
          
            spectrogram_np = spectrogram_np[:, None, ...]  
        waveforms = []
        for i in range(spectrogram_np.shape[0]):
       
            mag = spectrogram_np[i, 0]
            waveform = librosa.griffinlim(mag, n_iter=32, hop_length=hop_length, win_length=n_fft)
            waveforms.append(waveform)
            Fine_tune_logger.debug(f"Sample {i+1}: Waveform length after Griffin-Lim: {len(waveform)}")
        Fine_tune_logger.debug("Spectrogram to waveform conversion completed.")
        return waveforms






    def mask_loss(self,Fine_tune_logger, predicted_mask, mixture, target):
        Fine_tune_logger.debug("Calculating mask loss.")
        if predicted_mask.size() != mixture.size():
            raise ValueError(f"Shape mismatch: predicted_mask {predicted_mask.size()} and mixture {mixture.size()}!")
        predicted_vocals = predicted_mask * mixture
        if predicted_vocals.size() != target.size():
            raise ValueError(f"Shape mismatch: predicted_vocals {predicted_vocals.size()} and target {target.size()}!")
        l1 = self.l1_loss(predicted_vocals, target)
        stft = self.mse_loss(torch.log1p(predicted_vocals), torch.log1p(target))
        total_mask_loss = 0.5 * l1 + 0.5 * stft
        return total_mask_loss





    def perceptual_loss(self, Fine_tune_logger, output, target):
        import numpy as np
        import librosa
        Fine_tune_logger.debug("Calculating perceptual loss.")
        output_waveforms = self.spectrogram_to_waveform(output)
        target_waveforms = self.spectrogram_to_waveform(target)
        batch_size = len(output_waveforms)
        Fine_tune_logger.debug(f"Batch size for perceptual loss: {batch_size}")

        orig_sr, new_sr = 44100, 16000
        max_length = 176400  

        processed_output = []
        processed_target = []
        for i in range(batch_size):
            out_np = output_waveforms[i]
            tgt_np = target_waveforms[i]

            out_16k = librosa.resample(out_np, orig_sr=orig_sr, target_sr=new_sr)
            tgt_16k = librosa.resample(tgt_np, orig_sr=orig_sr, target_sr=new_sr)

            if len(out_16k) < max_length:
                out_16k = np.pad(out_16k, (0, max_length - len(out_16k)))
            else:
                out_16k = out_16k[:max_length]
            if len(tgt_16k) < max_length:
                tgt_16k = np.pad(tgt_16k, (0, max_length - len(tgt_16k)))
            else:
                tgt_16k = tgt_16k[:max_length]
            processed_output.append(out_16k)
            processed_target.append(tgt_16k)


        processed_output = self.preprocess_for_vggish(processed_output, fs=new_sr)
        processed_target = self.preprocess_for_vggish(processed_target, fs=new_sr)


        with torch.no_grad():
            try:
                output_features = self.audio_model(processed_output, fs=new_sr)
                target_features = self.audio_model(processed_target, fs=new_sr)
                Fine_tune_logger.debug("Batched VGGish feature extraction succeeded.")
            except Exception as e:

                Fine_tune_logger.error(f"Batched VGGish feature extraction failed: {e}. Falling back to per-sample extraction.")
                output_features_list, target_features_list = [], []
                for i in range(batch_size):
                    try:
                        out_feat = self.audio_model(processed_output[i].unsqueeze(0), fs=new_sr)
                        tgt_feat = self.audio_model(processed_target[i].unsqueeze(0), fs=new_sr)
                        output_features_list.append(out_feat)
                        target_features_list.append(tgt_feat)
                    except Exception as e:
                        Fine_tune_logger.error(f"Error extracting VGGish features for sample {i+1}: {e}")
                        continue
                if not output_features_list or not target_features_list:
                    Fine_tune_logger.warning("No features extracted for perceptual loss. Returning zero loss.")
                    return torch.tensor(0.0, device=self.device)
                output_features = torch.cat(output_features_list, dim=0)
                target_features = torch.cat(target_features_list, dim=0)

        loss_value = self.mse_loss(output_features, target_features)
        Fine_tune_logger.debug(f"Perceptual Loss value: {loss_value.item():.6f}")
        return loss_value





    def normalize_waveform(self, Fine_tune_logger, waveform):
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            normalized = waveform / max_val
            if torch.isnan(normalized).any() or torch.isinf(normalized).any():
                Fine_tune_logger.warning("NaN or Inf detected after normalization; returning zeros.")
                return torch.zeros_like(waveform)
            return normalized
        else:
            return torch.zeros_like(waveform)





    def spectral_loss(self, Fine_tune_logger, output, target, n_fft=1024, hop_length=512):
        Fine_tune_logger.debug("Calculating spectral loss.")
        output = self.normalize_waveform(output)
        target = self.normalize_waveform(target)
        if output.ndim == 4:
            output = output.squeeze(1)
            target = target.squeeze(1)
        elif output.ndim != 3:
            raise ValueError(f"Unexpected tensor dimensions for spectral loss. Output shape: {output.shape}, Target shape: {target.shape}")

        output_log = torch.log1p(torch.abs(output))
        target_log = torch.log1p(torch.abs(target))
        loss = self.mse_loss(output_log, target_log)
        Fine_tune_logger.debug(f"Spectral loss calculated: {loss.item():.6f}")
        return loss





    def multi_scale_loss(self, Fine_tune_logger, output, target, scales=[1, 2, 4]):
        import torch.nn.functional as F
        Fine_tune_logger.debug("Calculating multi-scale loss.")
        total_loss = 0.0
        for scale in scales:
            scaled_output = F.avg_pool2d(output, kernel_size=scale)
            scaled_target = F.avg_pool2d(target, kernel_size=scale)
            loss = self.mse_loss(scaled_output, scaled_target)
            total_loss += loss
            Fine_tune_logger.debug(f"Scale {scale}: MSE loss={loss.item():.6f}")
        Fine_tune_logger.debug(f"Total multi-scale loss: {total_loss.item():.6f}")
        return total_loss




    def forward(self,Fine_tune_logger, predicted_mask, mixture, target, outputs):
        Fine_tune_logger.debug("Forward pass for combined loss calculation.")
        mask_loss_val = self.mask_loss(predicted_mask, mixture, target)
        l1_val = self.l1_loss(outputs, target)
        mse_val = self.mse_loss(outputs, target)
        spectral_val = self.spectral_loss(outputs, target)
        perceptual_val = self.perceptual_loss(outputs, target)
        multi_scale_val = self.multi_scale_loss(outputs, target)

        total_loss = (
            0.3 * l1_val +
            0.3 * mse_val +
            0.1 * spectral_val +
            0.2 * perceptual_val +
            0.1 * multi_scale_val
        )
        Fine_tune_logger.debug(f"Total loss (weighted sum): {total_loss.item():.6f}")

        combined_loss = 0.5 * mask_loss_val + 0.5 * total_loss
        Fine_tune_logger.debug(f"Combined loss (mask + total): {combined_loss.item():.6f}")
        return combined_loss, mask_loss_val, total_loss






