import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
import torch
from torch.utils.tensorboard import SummaryWriter
from Training.Externals.Value_storage import writer_loss_batches, writer_loss_epoch


writer = SummaryWriter(log_dir="logs")

def log_batch_losses(global_step, combined_loss, mask_loss, hybrid_loss):
    writer_loss_batches[global_step] = {
        'combined': combined_loss.item(),
        'mask': mask_loss.item(),
        'hybrid': hybrid_loss.item()
    }
    writer.add_scalar("Loss/Batch/Combined", combined_loss.item(), global_step)
    writer.add_scalar("Loss/Batch/Mask",     mask_loss.item(),     global_step)
    writer.add_scalar("Loss/Batch/Hybrid",   hybrid_loss.item(),   global_step)



def log_epoch_losses(epoch, combined_loss_avg, mask_loss_avg, hybrid_loss_avg):
    writer_loss_epoch[epoch] = {
        'combined_epoch_avg': combined_loss_avg,
        'mask_epoch_avg': mask_loss_avg,
        'hybrid_epoch_avg': hybrid_loss_avg
    }
    writer.add_scalar("Loss/Epoch/Combined", combined_loss_avg, epoch)
    writer.add_scalar("Loss/Epoch/Mask",     mask_loss_avg,     epoch)
    writer.add_scalar("Loss/Epoch/Hybrid",   hybrid_loss_avg,   epoch)



def logging_avg_loss_epoch(epoch,prev_epoch_loss,epochs,avg_epoch_loss,maskloss_avg,hybridloss_avg,train_logger):
    train_logger.info(
    f"[Epoch Summary] Epoch: {epoch + 1}/{epochs}, "
    f"Avg Combined Loss: {avg_epoch_loss:.6f}, MaskLoss: {maskloss_avg:.6f}, "
    f"Hybridloss: {hybridloss_avg:.6f}"
    f"Previous Epoch loss: {prev_epoch_loss}"
    )



def logging_avg_loss_batches(train_logger,epoch,epochs,avg_combined_loss,avg_mask_loss,avg_hybrid_loss):
    train_logger.info(
    f"[Validation] Epoch {epoch + 1}/{epochs}: "
    f"Combined Loss={avg_combined_loss:.6f}, "
    f"Mask Loss={avg_mask_loss:.6f}, "
    f"Hybrid Loss={avg_hybrid_loss:.6f}"
    )



def prev_epoch_loss_log(train_logger,prev_epoch_loss,avg_epoch_loss,epoch):
        if prev_epoch_loss is not None:  
            loss_improvement = (prev_epoch_loss - avg_epoch_loss) / prev_epoch_loss * 100
            train_logger.info( f"[Epoch Improvement] Epoch {epoch + 1}: Loss improved by {loss_improvement:.2f}% from previous epoch." )
        else:
            train_logger.info(f"[Epoch Improvement] Epoch {epoch + 1}: No comparison (first epoch).")

        


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


