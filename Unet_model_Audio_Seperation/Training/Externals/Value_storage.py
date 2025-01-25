import os
import sys

# Setup logger
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Training.Externals.utils import Return_root_dir
from Training.Externals.Logger import setup_logger
from Training.Externals.utils import Return_root_dir

root_dir = Return_root_dir() #Gets the root directory
train_log_path = os.path.join(root_dir, "Model_performance_logg/log/Model_Training_logg.txt")
root_dir = Return_root_dir() #Gets the root directory
train_logger = setup_logger('train',train_log_path)

global loss_history_Epoches
global loss_history_Batches
# Define loss history dictionaries
loss_history_Epoches = {
    "mask_loss": [],        # Mask loss per epoch
    "hybrid_loss": [],      # Hybrid loss per epoch
    "combined": [],         # Combined loss (weighted sum of mask loss and hybridloss) per epoch
    "Total_loss_per_epoch": [],  # Total loss per epoch
}

loss_history_Batches = {
    "mask_loss": [],        # Mask loss per batch
    "hybrid_loss": [],      # Hybrid loss per batch
    "combined": [],         # Combined loss per batch
}


def Append_loss_values_for_batch(mask_loss, hybrid_loss, combined_loss):
    train_logger = setup_logger('train',train_log_path)
    train_logger.info(f"Appending Batch Losses: Mask: {mask_loss.item()}, Hybrid: {hybrid_loss.item()}, Combined: {combined_loss.item()}")
    loss_history_Batches["mask_loss"].append(mask_loss.item())
    loss_history_Batches["hybrid_loss"].append(hybrid_loss.item())
    loss_history_Batches["combined"].append(combined_loss.item())
    train_logger.debug(f"Current batch loss history (last 3): {loss_history_Batches['mask_loss'][-3:]}, " f"{loss_history_Batches['hybrid_loss'][-3:]}, {loss_history_Batches['combined'][-3:]}")
    print(f"Mask Loss: {loss_history_Batches['mask_loss'][-3:]}")
    print(f"Hybrid Loss: {loss_history_Batches['hybrid_loss'][-3:]}")
    print(f"Combined Loss: {loss_history_Batches['combined'][-3:]}")
    


def Append_loss_values_for_epoches(mask_loss_avg, hybrid_loss_avg, combined_loss_avg, avg_epoch_loss):
    train_logger
    loss_history_Epoches["mask_loss"].append(mask_loss_avg)
    loss_history_Epoches["hybrid_loss"].append(hybrid_loss_avg)
    loss_history_Epoches["combined"].append(combined_loss_avg)
    loss_history_Epoches["Total_loss_per_epoch"].append(avg_epoch_loss)
    train_logger.info(f"Epoch summary - Mask Loss Avg: {mask_loss_avg:.4f}, Hybrid Loss Avg: {hybrid_loss_avg:.4f}, " f"Combined Loss Avg: {combined_loss_avg:.4f}, Total Loss for Epoch: {avg_epoch_loss:.4f}")
    print(f"Mask Loss: {loss_history_Epoches['mask_loss'][-3:]}")
    print(f"Hybrid Loss: {loss_history_Epoches['hybrid_loss'][-3:]}")
    print(f"Combined Loss: {loss_history_Epoches['combined'][-3:]}")
    print(f"Combined Loss: {loss_history_Epoches['Total_loss_per_epoch'][-3:]}")


def Get_calculated_average_loss_from_batches():
    if len(loss_history_Batches["mask_loss"]) > 0:
        mask_loss_avg = sum(loss_history_Batches["mask_loss"]) / len(loss_history_Batches["mask_loss"])
        hybrid_loss_avg = sum(loss_history_Batches["hybrid_loss"]) / len(loss_history_Batches["hybrid_loss"])
        combined_loss_avg = sum(loss_history_Batches["combined"]) / len(loss_history_Batches["combined"])
        train_logger.debug(f"Calculated averages for batch losses: Mask Loss Avg: {mask_loss_avg:.4f}, " f"Hybrid Loss Avg: {hybrid_loss_avg:.4f}, Combined Loss Avg: {combined_loss_avg:.4f}")
        print(f"Calculated averages for batch losses: Mask Loss Avg: {mask_loss_avg:.4f}, " f"Hybrid Loss Avg: {hybrid_loss_avg:.4f}, Combined Loss Avg: {combined_loss_avg:.4f}")
    else:
        mask_loss_avg = hybrid_loss_avg = combined_loss_avg = 0.0
        train_logger.warning("No batch losses recorded yet. Returning 0 averages.")

    return mask_loss_avg, hybrid_loss_avg, combined_loss_avg
