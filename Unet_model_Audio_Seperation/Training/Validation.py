import torch
import os
from torch import autocast
from Training.Externals.Functions import Return_root_dir
root_dir = Return_root_dir()
from Training.Externals.Logger import setup_logger
from Training.Externals.Functions import Early_break
validation_log_path = os.path.join(root_dir, "Model_Performance_logg/log/Validation_logg.txt")
Validation_logger = setup_logger('Validation', validation_log_path)


bestloss = float('inf')
trigger_times = 0 
patience = 5

def Validate_ModelEngine(epoch, Model_Engine, combined_val_loader, criterion, Model_CheckPoint, current_step):
    global bestloss, trigger_times 
    Validation_logger.info(f"Validation Size: [{len(combined_val_loader)}]")
    device = next(Model_Engine.parameters()).device
    Model_Engine.to(device)
    Model_Engine.eval()
    
    val_running_loss = 0.0
    with torch.no_grad(), autocast(device_type='cuda', enabled=(device.type == 'cuda')):
        for val_batch_idx, (val_inputs, val_targets) in enumerate(combined_val_loader):
            if val_batch_idx <= 3:
              Validation_logger.info(f"val_batch_idx: [{val_batch_idx}], inputs: [{val_inputs}], target: [{val_targets}]")
            val_inputs = val_inputs.to(device,torch.float16, non_blocking=True)
            val_targets = val_targets.to(device,torch.float16, non_blocking=True)

            val_predicted_mask, val_outputs = Model_Engine(val_inputs)
            val_combined_loss, *_ = criterion(val_predicted_mask, val_inputs, val_targets)

            val_running_loss += val_combined_loss.item()

    avg_validation_loss = val_running_loss / len(combined_val_loader)
    Validation_logger.info(f"[Epoch {epoch + 1}] Validation Loss: [{avg_validation_loss:.6f}], runningloss: [{val_running_loss}],val_combined_loss: [{val_combined_loss.item()}], val_predicted_mask: [{val_predicted_mask.item()}], val_outputs: [{val_outputs.shape}]")

    # Model checkpointing logic
    if avg_validation_loss < bestloss:
        Validation_logger.info(f"New best model (Loss: {avg_validation_loss:.6f} < {bestloss:.6f})")
        bestloss = avg_validation_loss
        checkpoint_dir = os.path.join(Model_CheckPoint, f"checkpoint_epoch_{epoch + 1}")
        
        try:
            Model_Engine.save_checkpoint(checkpoint_dir, {
                'step': current_step,
                'best_loss': bestloss,
                'trigger_times': trigger_times
            })
            trigger_times = 0
        except Exception as e:
            Validation_logger.error(f"Checkpoint save failed: {str(e)}")
    else:
        trigger_times += 1
        Validation_logger.info(f"No improvement (Trigger: {trigger_times}/{patience})")

  
    if Early_break(trigger_times, patience):
        Validation_logger.warning(f"Early stopping triggered at epoch {epoch + 1}")

    return bestloss, trigger_times, avg_validation_loss