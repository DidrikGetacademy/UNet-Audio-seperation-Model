
import torch
import os
import sys
import torch.nn as nn
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Training.Externals.Logger import setup_logger
from Training.Externals.utils import Return_root_dir
root_dir = Return_root_dir() #Gets the root directory
train_log_path = os.path.join(root_dir, "Model_performance_logg/log/Model_Training_logg.txt")

train_logger = setup_logger( 'train',train_log_path)


#MASK-ESTIMATION-LOSS CLASS
class MaskEstimationLoss(nn.Module):
    def __init__(self):
        super(MaskEstimationLoss,self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted_mask, mixture, target):
            device = predicted_mask.device  # Log the device of the tensors
            print(f"[MaskEstimationLoss] Moving tensors to {device}...")

            #Estimated vocal spectrum from mask
            predicted_vocals = predicted_mask * mixture
            print(f"[MaskEstimationLoss] Predicted vocals on device {device}")
            #l1 loss
            l1 = self.l1_loss(predicted_vocals, target)


            stft = self.mse_loss(torch.log1p(predicted_vocals), torch.log1p(target))

            return 0.5 * l1 + 0.5 * stft


#COMBINED LOSS CLASS
class Combinedloss(nn.Module):
    def __init__(self):
        super(Combinedloss, self).__init__()
        self.mask_loss = MaskEstimationLoss()
        self.hybrid_loss = HybridLoss()

    def forward(self, predicted_mask, mixture, target):
        device = predicted_mask.device  # Log the device of the tensors
        print(f"[Combinedloss] Predicted mask and target on device {device}")
        # Calculate the mask loss
        mask_loss = self.mask_loss(predicted_mask, mixture, target)
        print(f"[Combinedloss] Mask loss on device {device}: {mask_loss.item():.6f}")

        # Estimated vocals (applying mask to mixture)
        predicted_vocals = predicted_mask * mixture

        # Calculate the hybrid loss on the predicted vocals
        hybrid_loss, l1_loss, stft_loss = self.hybrid_loss(predicted_vocals, target)
        print(f"[Combinedloss] Hybrid loss: {hybrid_loss.item():.6f}, L1 loss: {l1_loss.item():.6f}, STFT loss: {stft_loss.item():.6f}")

        # Combine the losses
        combined_loss = 0.5 * mask_loss + 0.5 * hybrid_loss

        return combined_loss, mask_loss, hybrid_loss



        

#HYBRID-LOSS CLASS
class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        device = pred.device  
        pred = pred.float()
        target = target.float()
        print(f"[HybridLoss] Pred and Target on device {device}")


        l1 = self.l1_loss(pred, target)
        n_fft = min(1024, pred.size(-1))
        hop_length = 512
        window = torch.hann_window(n_fft, device=device)
        batch_size = pred.size(0)
        stft_loss = 0.0
        for i in range(batch_size):
            pred_tensor = pred[i].squeeze()
            target_tensor = target[i].squeeze()
            min_len = min(pred_tensor.size(-1), target_tensor.size(-1))
            pred_tensor = pred_tensor[..., :min_len]
            target_tensor = target_tensor[..., :min_len]
            pred_stft = torch.stft(pred_tensor, n_fft=n_fft, hop_length=hop_length, return_complex=True, window=window)
            target_stft = torch.stft(target_tensor, n_fft=n_fft, hop_length=hop_length, return_complex=True, window=window)
            pred_stft_log = torch.log1p(torch.abs(pred_stft))
            target_stft_log = torch.log1p(torch.abs(target_stft))
            stft_loss_batch = self.mse_loss(pred_stft_log, target_stft_log)
            stft_loss += stft_loss_batch
            train_logger.debug(f"[HybridLoss] Batch {i}: STFT MSE={stft_loss_batch.item():.6f}")
        stft_loss /= batch_size
        combined_loss = 0.3 * l1 + 0.7 * stft_loss
        train_logger.debug(f"[HybridLoss] L1={l1.item():.6f}, STFT={stft_loss.item():.6f}, Combined={combined_loss.item():.6f}")
        return combined_loss, l1, stft_loss





