import torch
import os
from torch import autocast
from Training.Externals.Functions import ( Return_root_dir,)
root_dir = Return_root_dir()
import sys
from Training.Externals.Logger import setup_logger
from Training.Externals.Functions import Early_break
validation_log_path = os.path.join(root_dir, "Evaluation/Log/Evaluation_logg.txt")
Validation_logger = setup_logger('Validation', validation_log_path)
patience = 5

def Validate_ModelEngine(epoch, Model_Engine,combined_val_loader,criterion, bestloss, Model_CheckPoint,trigger_times,current_step):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Model_Engine.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for val_batch_idx, (val_inputs, val_targets) in enumerate(combined_val_loader):
            val_inputs = val_inputs.to(device,non_blocking=True)
            val_targets = val_targets.to(device,non_blocking=True)

        with autocast(device_type='cuda',enbaled=(device.type == 'cuda')):
            val_predicted_mask , val_outputs = Model_Engine(val_inputs)
            val_combined_loss, _, _ = criterion(val_predicted_mask, val_inputs, val_targets)


            val_running_loss += val_combined_loss.item()


            avg_validation_loss = val_running_loss / len(combined_val_loader)
            Validation_logger.info(f"[Epoch {epoch + 1}] Validation Loss: {avg_validation_loss:.6f}")

            if avg_validation_loss < bestloss:
                print("New best epoch loss! saving model now")
                bestloss = avg_validation_loss
                Validation_logger.info(f"Avg_epoch_loss={avg_validation_loss}, bestloss= {bestloss}")
                checkpoint_dir = os.path.join(Model_CheckPoint, f"checkpoint_epochsss_{epoch + 1}")
                client_sd = {'step': current_step, 'best_loss': bestloss, 'trigger_times': trigger_times}      
                try:
                    Model_Engine.save_checkpoint(checkpoint_dir,client_sd)
                    Validation_logger.info(f"New best model saved to {checkpoint_dir}, [Epoch {epoch + 1}] with client_sd: {client_sd} with validation loss {avg_validation_loss:.6f}")
                except Exception as e: 
                    Validation_logger.error(f"Error while saving checkpoint: {e}")
            else:
                trigger_times += 1
                Validation_logger.info(f"NO IMPROVEMENT IN LOSS!. Previous best epoch loss: {bestloss:.6f}, This epoch's loss: {avg_validation_loss:.6f}, trigger times: {trigger_times}")                      

                    
            if Early_break(trigger_times, patience):
                print(f"[Epoch {epoch + 1}] Early stopping triggered.")
                sys.exit(0)


