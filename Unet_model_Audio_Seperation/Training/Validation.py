import torch
import os
from torch import autocast
from Training.Externals.Functions import Return_root_dir
root_dir = Return_root_dir()
from Training.Externals.Logger import setup_logger
from Training.Externals.Functions import Early_break

validation_log_path = os.path.join(root_dir, "Model_Performance_logg/log/Validation_logg.txt")
Validation_logger = setup_logger('Validation', validation_log_path)

trigger_times = 0 
patience = 5

def Validate_ModelEngine(epoch, Model_Engine, combined_val_loader, criterion, Model_CheckPoint, current_step):
    Validation_logger.info(f"Validation Size: [{len(combined_val_loader)}]")
    device = next(Model_Engine.parameters()).device
    Model_Engine.to(device)
    Model_Engine.eval()
    Model_Engine.to(dtype=torch.bfloat16)
    
    val_running_loss = 0.0
    with torch.no_grad():
        for val_batch_idx, (val_inputs, val_targets) in enumerate(combined_val_loader):
            val_inputs = val_inputs.to(device, non_blocking=True)
            val_targets = val_targets.to(device, non_blocking=True)

            if val_batch_idx <= 2:
                Validation_logger.info(f"VALIDATION SHAPE CHECK ---> Val_targets: [{val_targets.shape}], Val_inputs: [{val_inputs.shape}]")

            with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                if val_batch_idx <= 2:
                    Validation_logger.info(f"VALIDATION input for model: {val_inputs.shape}")

                val_predicted_mask, val_outputs = Model_Engine(val_inputs)
                val_combined_loss, _, _ = criterion(val_predicted_mask, val_inputs, val_targets)


                val_running_loss += val_combined_loss.item()

 
        avg_validation_loss = val_running_loss / len(combined_val_loader)
        Validation_logger.info(f"[Epoch {epoch + 1}] Validation Loss: {avg_validation_loss:.6f}")

   
        global trigger_times
        if avg_validation_loss < bestloss:
            print("New best epoch loss! Saving model now")
            bestloss = avg_validation_loss
            Validation_logger.info(f"Avg_epoch_loss={avg_validation_loss}, bestloss= {bestloss}")
            checkpoint_dir = os.path.join(Model_CheckPoint, f"checkpoint_epochsss_{epoch + 1}")
            client_sd = {'step': current_step, 'best_loss': bestloss, 'trigger_times': trigger_times}      
            try:
                Model_Engine.save_checkpoint(checkpoint_dir, client_sd)
                Validation_logger.info(f"New best model saved to {checkpoint_dir}, [Epoch {epoch + 1}] with client_sd: {client_sd} with validation loss {avg_validation_loss:.6f}")
                trigger_times = 0  
            except Exception as e: 
                Validation_logger.error(f"Error while saving checkpoint: {e}")
        else:
            trigger_times += 1
            Validation_logger.info(f"NO IMPROVEMENT IN LOSS! Previous best epoch loss: {bestloss:.6f}, This epoch's loss: {avg_validation_loss:.6f}, trigger times: {trigger_times}") 

        # Early stopping
        if Early_break(trigger_times, patience):
            print(f"[Epoch {epoch + 1}] Early stopping triggered.")

    return bestloss, trigger_times, avg_validation_loss  