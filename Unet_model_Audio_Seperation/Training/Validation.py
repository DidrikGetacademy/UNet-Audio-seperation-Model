import torch
import os
from torch import autocast
from Training.Externals.Functions import Return_root_dir
root_dir = Return_root_dir()
from Training.Externals.Logger import setup_logger
from Training.Externals.Functions import Early_break
validation_log_path = os.path.join(root_dir, "Model_Performance_logg/log/Validation_logg.txt")
Validation_logger = setup_logger('Validation', validation_log_path)
from Datasets.Scripts.Dataset_utils import Convert_spectrogram_to_audio


bestloss_validation = float('inf')
trigger_times = 0 
patience = 5

def Validate_ModelEngine(epoch, Model_Engine, combined_val_loader, criterion, Model_CheckPoint, current_step):
    global bestloss_validation, trigger_times 
    Validation_logger.info(f"Validation Size: [{len(combined_val_loader)}]")
    device = next(Model_Engine.parameters()).device
    Model_Engine.to(device)
    Model_Engine.eval()
    
    val_running_loss = 0.0
    with torch.no_grad(), autocast(device_type='cuda', enabled=(device.type == 'cuda')):
        for val_batch_idx, (val_inputs, val_targets) in enumerate(combined_val_loader):
            if val_inputs is None or val_targets is None or val_inputs.numel() == 0 or val_targets.numel() == 0:  
                       Validation_logger.warning(f"Skipping batch {val_batch_idx} due to empty inputs or targets.")
                       continue  
            if val_batch_idx <= 3:
              Validation_logger.info(f"val_batch_idx: [{val_batch_idx}], inputs: [{val_inputs.shape}], target: [{val_targets.shape}]")
            val_inputs = val_inputs.to(device,torch.float16, non_blocking=True)
            val_targets = val_targets.to(device,torch.float16, non_blocking=True)

            val_predicted_mask, val_outputs = Model_Engine(val_inputs)
            predicted_vocals = val_predicted_mask * val_inputs
            if val_batch_idx <= 2:  
                Convert_spectrogram_to_audio(audio_path="/mnt/c/Users/didri/Desktop/Programmering/ArtificalintelligenceModels/UNet-Model_Vocal_Isolation/Unet_model_Audio_Seperation/audio_logs/Validering",predicted_vocals=predicted_vocals[0],targets = val_targets[0],inputs=val_inputs[0],outputs=None)
            val_combined_loss, mask_loss, hybrid_loss_val, l1_loss_val, stft_loss_val, sdr_loss = criterion(val_predicted_mask, val_inputs, val_targets)

            val_running_loss += val_combined_loss.item()

    avg_validation_loss = val_running_loss / len(combined_val_loader)
    Validation_logger.info(
        f"[VALIDATION] Epoch {epoch + 1}, \n Validation Loss: [{avg_validation_loss:.6f}], \n"
        f"Running Loss: [{val_running_loss:.6f}], \n"
        f"Combined Validation Loss: [{val_combined_loss.item():.6f}],\n "
        f"Predicted Mask Shape: [{val_predicted_mask.shape}],\n "
        f"Predicted Mask Mean: [{val_predicted_mask.mean().item():.6f}], \n"
        f"Outputs Shape: [{val_outputs.shape}], \n"
        f"hybrid_loss_val [{hybrid_loss_val.item():.6f}]\n"
        f"mask_loss [{mask_loss.item():.6f}]\n"
        f"stft_loss_val[{stft_loss_val.item():.6f}]\n"
        f"l1_loss_val[{l1_loss_val.item():.6f}]\n"
        f"SDR Loss: [{sdr_loss.item():.6f}]\n"
    )


    # Model checkpointing logic
    if avg_validation_loss < bestloss_validation:
        Validation_logger.info(f"New best model (Loss: {avg_validation_loss:.6f} < {bestloss_validation:.6f})")
        bestloss_validation = avg_validation_loss
        checkpoint_dir = os.path.join(Model_CheckPoint, f"checkpoint_epoch_{epoch + 1}")
        
        try:
            Model_Engine.save_checkpoint(checkpoint_dir, {
                'step': current_step,
                'best_loss': bestloss_validation,
                'trigger_times': trigger_times
            })
            Validation_logger.info(f"Checkpoint saved at: {checkpoint_dir}")
            trigger_times = 0
        except Exception as e:
            Validation_logger.error(f"Checkpoint save failed: {str(e)}")
    else:
        trigger_times += 1
        Validation_logger.info(f"No improvement (Trigger: {trigger_times}/{patience})")

  
    if Early_break(trigger_times, patience):
         Validation_logger.warning(f"Early stopping triggered at epoch {epoch + 1}")
         raise ValueError(f"trigger times {trigger_times}, patience {patience} Validation")

    return bestloss_validation, trigger_times, avg_validation_loss