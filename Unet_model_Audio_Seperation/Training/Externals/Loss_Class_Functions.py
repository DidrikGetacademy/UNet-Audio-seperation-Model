from Training.Externals.Logger import setup_logger
import torch
import os
import sys
import torch.nn as nn
from Training.Externals.Logger import setup_logger
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
train_logger = setup_logger( 'train', r'C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_performance_logg\log\Model_Training_logg.txt')


#MASK-ESTIMATION-LOSS CLASS
class MaskEstimationLoss(nn.Module):
    def __init__(self):
        super(MaskEstimationLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted_mask, mixture, target):
        # Ensure tensors are on the same device
        device = predicted_mask.device
        mixture = mixture.to(device)
        target = target.to(device)

        # Estimate vocal spectrum from the mask
        predicted_vocals = predicted_mask * mixture

        # L1 loss
        l1 = self.l1_loss(predicted_vocals, target)

        # STFT-based loss
        stft = self.mse_loss(torch.log1p(predicted_vocals), torch.log1p(target))

        # Combine the losses (equal weights)
        return 0.5 * l1 + 0.5 * stft




#COMBINED LOSS CLASS
class Combinedloss(nn.Module):
    def __init__(self):
        super(Combinedloss, self).__init__()
        self.mask_loss = MaskEstimationLoss()
        self.hybrid_loss = HybridLoss()

    def forward(self, predicted_mask, mixture, target):
        # Ensure tensors are on the same device
        device = predicted_mask.device
        mixture = mixture.to(device)
        target = target.to(device)

        # Calculate the mask loss
        mask_loss = self.mask_loss(predicted_mask, mixture, target)

        # Estimate vocals (applying mask to mixture)
        predicted_vocals = predicted_mask * mixture

        # Calculate the hybrid loss on the predicted vocals
        hybrid_loss, l1_loss, stft_loss = self.hybrid_loss(predicted_vocals, target)

        # Combine the losses (equal weights)
        combined_loss = 0.5 * mask_loss + 0.5 * hybrid_loss

        return combined_loss, mask_loss, hybrid_loss


        

#HYBRID-LOSS CLASS
class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.n_fft = 2048  # Maximum FFT size
        self.hop_length = 512
        # Precompute the Hann window and register it as a buffer (automatically moves with the model's device)
        self.register_buffer('window', torch.hann_window(self.n_fft))

    def forward(self, pred, target):
        # Ensure the Hann window is on the same device as the predictions
        window = self.window.to(pred.device)

        # Convert predictions and targets to float (ensuring precision)
        pred = pred.float()
        target = target.float()

        # Calculate L1 loss
        l1 = self.l1_loss(pred, target)

        # Determine the FFT size dynamically based on the input length
        n_fft = min(self.n_fft, pred.size(-1))
        batch_size = pred.size(0)

        stft_loss = 0.0
        for i in range(batch_size):
            # Extract the current batch's prediction and target tensors
            pred_tensor = pred[i].squeeze()
            target_tensor = target[i].squeeze()

            # Ensure tensors have the same length by truncating to the shorter size
            min_len = min(pred_tensor.size(-1), target_tensor.size(-1))
            pred_tensor = pred_tensor[..., :min_len]
            target_tensor = target_tensor[..., :min_len]

            # Compute STFT for both prediction and target
            pred_stft = torch.stft(pred_tensor, n_fft=n_fft, hop_length=self.hop_length, return_complex=True, window=window)
            target_stft = torch.stft(target_tensor, n_fft=n_fft, hop_length=self.hop_length, return_complex=True, window=window)

            # Compute log-scaled STFT magnitudes
            pred_stft_log = torch.log1p(torch.abs(pred_stft))
            target_stft_log = torch.log1p(torch.abs(target_stft))

            # Compute MSE loss between log-scaled STFTs
            stft_loss_batch = self.mse_loss(pred_stft_log, target_stft_log)
            stft_loss += stft_loss_batch

            # Log debug information for each batch
            train_logger.debug(f"[HybridLoss] Batch {i}: STFT MSE={stft_loss_batch.item():.6f}")

        # Compute average STFT loss across the batch
        stft_loss /= batch_size

        # Combine L1 and STFT losses
        combined_loss = 0.3 * l1 + 0.7 * stft_loss

        # Log debug information for combined loss
        train_logger.debug(f"[HybridLoss] L1={l1.item():.6f}, STFT={stft_loss.item():.6f}, Combined={combined_loss.item():.6f}")

        return combined_loss, l1, stft_loss

