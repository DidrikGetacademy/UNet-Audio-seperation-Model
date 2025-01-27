import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)

from Training.Externals.utils import Return_root_dir   
root_dir = Return_root_dir()





def logging_avg_loss_epoch(epoch, prev_epoch_loss, epochs, avg_epoch_loss, maskloss_avg, hybridloss_avg, train_logger):
    train_logger.info(
    f"[Epoch Summary] Epoch: {epoch + 1}/{epochs}, "
    f"Avg Combined Loss: {avg_epoch_loss:.6f}, MaskLoss: {maskloss_avg:.6f}, "
    f"Hybridloss: {hybridloss_avg:.6f}"
    f"Previous Epoch loss: {prev_epoch_loss}"
    )


#Ikke i bruk 
def logging_avg_loss_batches(train_logger,epoch,epochs,avg_combined_loss,avg_mask_loss,avg_hybrid_loss):
    train_logger.info(
    f"[Validation] Epoch {epoch + 1}/{epochs}: "
    f"Combined Loss={avg_combined_loss:.6f}, "
    f"Mask Loss={avg_mask_loss:.6f}, "
    f"Hybrid Loss={avg_hybrid_loss:.6f}"
    )



def prev_epoch_loss_log(train_logger,prev_epoch_loss,avg_epoch_loss,epoch):
        if prev_epoch_loss is None:
            prev_epoch_loss = avg_epoch_loss  
        if prev_epoch_loss is not None:  
            loss_improvement = (prev_epoch_loss - avg_epoch_loss) / prev_epoch_loss * 100
            train_logger.info( f"[Epoch Improvement] Epoch {epoch + 1}: Loss improved by {loss_improvement:.2f}% from previous epoch." )
        else:
            train_logger.info(f"[Epoch Improvement] Epoch {epoch + 1}: No comparison (first epoch).")





def log_first_2_batches_inputs_targets(batch_idx,train_logger,inputs,targets):
        if batch_idx < 2:    
            train_logger.info(f"Batch {batch_idx}: Mixture shape={inputs.shape}, Target shape={targets.shape}")
            train_logger.info(f"Mixture min={inputs.min().item():.4f}, max={inputs.max().item():.4f}")
            train_logger.info(f"Target min={targets.min().item():.4f}, max={targets.max().item():.4f}")


def log_first_2_batches_outputs_inputs_targets_predicted_mask(batch_idx,outputs,inputs,targets,predicted_mask,train_logger):
         if batch_idx < 2:
            train_logger.info(f"Batch {batch_idx}: Mask range: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}")
            train_logger.debug(f"Batch {batch_idx}: Inputs shape={inputs.shape}, Targets shape={targets.shape}, Predicted Mask shape={predicted_mask.shape}, Outputs shape={outputs.shape}")
            train_logger.debug(f"Mask min={predicted_mask.min().item()}, max={predicted_mask.max().item()}")


import logging
def setup_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():  
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(file_handler)
    return logger


