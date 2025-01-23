import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)


writer_loss_batches = {}
writer_loss_epoch = {}

loss_history_Epoches = {
    "mask_loss": [],        # Mask loss per epoch
    "hybrid_loss": [],      # Hybrid loss per epoch
    "combined": [],      # Combined loss (weighted sum of mask loss and hybridloss) per epoch
    "Total_loss_per_epoch": [],  # Total loss per epoch
}

loss_history_Batches = {
    "mask_loss": [],     # Mask loss per batch
    "hybrid_loss": [],   # Hybrid loss per batch
    "combined": [],      # Combined loss per batch
}



def Append_loss_values_for_batch(mask_loss, hybrid_loss, combined_loss):
    loss_history_Batches["mask_loss"].append(mask_loss.item())
    loss_history_Batches["hybrid_loss"].append(hybrid_loss.item())
    loss_history_Batches["combined"].append(combined_loss.item())

def Append_loss_values_for_epoches(mask_loss_avg,hybrid_loss_avg,combined_loss_avg,avg_epoch_loss):
    loss_history_Epoches["mask_loss"].append(mask_loss_avg)
    loss_history_Epoches["hybrid_loss"].append(hybrid_loss_avg)
    loss_history_Epoches["combined"].append(combined_loss_avg)
    loss_history_Epoches["Total_loss_per_epoch"].append(avg_epoch_loss)


def Get_calculated_average_loss_from_batches():
    mask_loss_avg = sum(loss_history_Batches["mask_loss"]) / len(loss_history_Batches["mask_loss"])
    hybrid_loss_avg = sum(loss_history_Batches["hybrid_loss"]) / len(loss_history_Batches["hybrid_loss"])
    combined_loss_avg = sum(loss_history_Batches["combined"]) / len(loss_history_Batches["combined"])
    return mask_loss_avg, hybrid_loss_avg, combined_loss_avg
