import matplotlib.pyplot as plt
import sys
import os
import librosa
import torch
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Training.Externals.utils import Return_root_dir
from Training.Externals.Logger import setup_logger
root_dir = Return_root_dir() #Gets the root directory
train_log_path = os.path.join(root_dir, "Model_performance_logg/log/Model_Training_logg.txt")
train_logger = setup_logger('loss_diagram_values',train_log_path)
diagramdirectory = os.path.join(root_dir,"Model_performance_logg/Diagrams")
os.makedirs(diagramdirectory, exist_ok=True)
from mir_eval.separation import bss_eval_sources
from Datasets.Scripts.Dataset_utils import spectrogram_to_waveform



####TRAINING####
#### TRAINING ####

def create_loss_table_epoches(loss_history_Epochs, loss_logger):
    table_rows = []
    for epoch, (mask_avg, hybrid_avg, combined_avg, epoch_loss) in enumerate(
        zip(
            loss_history_Epochs["mask_loss"],
            loss_history_Epochs["hybrid_loss"],
            loss_history_Epochs["combined"],
            loss_history_Epochs["Total_loss_per_epoch"]
        ),
        start=1
    ):
        table_rows.append((epoch, mask_avg, hybrid_avg, combined_avg, epoch_loss))

    headers = ["Epoch", "Mask Loss Avg", "Hybrid Loss Avg", "Combined Loss Avg", "Epoch Loss"]
    loss_logger.info("=" * 80)
    loss_logger.info(f"{headers[0]:<5}  {headers[1]:<14}  {headers[2]:<16}  {headers[3]:<16}  {headers[4]:<10}")
    loss_logger.info("=" * 80)

    for (epoch, mask_avg, hybrid_avg, combined_avg, epoch_loss) in table_rows:
        row_str = f"{epoch:<5}  {mask_avg:<14.6f}  {hybrid_avg:<16.6f}  {combined_avg:<16.6f}  {epoch_loss:<10.6f}"
        loss_logger.info(row_str + "\n")


def create_loss_table_batches(loss_history_Batches, loss_logger):
    table_rows = []
    for batch, (mask_loss, hybrid_loss, combined_loss) in enumerate(
        zip(
            loss_history_Batches["mask_loss"],
            loss_history_Batches["hybrid_loss"],
            loss_history_Batches["combined"]
        ),
        start=1
    ):
        table_rows.append((batch, mask_loss, hybrid_loss, combined_loss))

    headers = ["Batch", "Mask Loss", "Hybrid Loss", "Combined Loss"]
    loss_logger.info("=" * 80)
    loss_logger.info(f"{headers[0]:<6}  {headers[1]:<10}  {headers[2]:<12}  {headers[3]:<14}")
    loss_logger.info("=" * 80)

    for (batch, mask_val, hybrid_val, combined_val) in table_rows:
        row_str = f"{batch:<6}  {mask_val:<10.6f}  {hybrid_val:<12.6f}  {combined_val:<14.6f}"
        loss_logger.info(row_str + "\n")





def create_loss_diagrams(loss_history_Batches, loss_history_Epoches,loss_logger):
    loss_logger.info("Starting diagram creation process.")

    # Check if loss history dictionaries have values
    if not any(len(v) > 0 for v in loss_history_Batches.values()):
        loss_logger.warning("Loss history for batches is empty. Skipping batch loss diagram creation.")
    else:
        batches_figpath = os.path.join(diagramdirectory, "loss_curves_training_batches.png")
        plot_loss_curves_Training_script_Batches(loss_history_Batches,loss_logger, out_path=batches_figpath)
        loss_logger.info(f"Batch loss diagram saved at {batches_figpath}.")

    if not any(len(v) > 0 for v in loss_history_Epoches.values()):
        loss_logger.warning("Loss history for epochs is empty. Skipping epoch loss diagram creation.")
    else:
        epoch_figpath = os.path.join(diagramdirectory, "loss_curves_training_epoches.png")
        plot_loss_curves_Training_script_epoches(loss_history_Epoches,loss_logger, out_path=epoch_figpath)
        loss_logger.info(f"Epoch loss diagram saved at {epoch_figpath}.")

    loss_logger.info("Diagram creation process completed.")







####TRAINING EPOCHES#####
def plot_loss_curves_Training_script_epoches(loss_history_Epoches,loss_logger, out_path="loss_curves_training_epoches.png"):
    epochs_count = len(loss_history_Epoches["Total_loss_per_epoch"])
    loss_logger.info(f"Epochs Count: {epochs_count}, Total Loss Entries: {len(loss_history_Epoches['Total_loss_per_epoch'])}")

    if len(loss_history_Epoches["Total_loss_per_epoch"]) != epochs_count:
        loss_logger.error(f"Epoch Loss Mismatch: Expected {epochs_count}, Found {len(loss_history_Epoches['Total_loss_per_epoch'])}.") 
        return  

    epochs = list(range(1, epochs_count + 1))
    

    y_min, y_max = float('inf'), float('-inf')

 
    for loss_name in ["mask_loss", "hybrid_loss", "combined", "Total_loss_per_epoch"]:
        loss_list = loss_history_Epoches.get(loss_name, [])
        if loss_list:  
            y_min = min(y_min, min(loss_list))
            y_max = max(y_max, max(loss_list))


    if y_min < float('inf') and y_max > float('-inf'):
        plt.ylim([y_min * 0.9, y_max * 1.1]) 

    plt.figure(figsize=(12, 6))


    for loss_name, label, color in [
        ("mask_loss", "mask_loss", "blue"),
        ("hybrid_loss", "hybrid_loss", "green"),
        ("combined", "Combined-loss", "yellow"),
        ("Total_loss_per_epoch", "Total_loss_per_epoch", "purple")
    ]:
        if len(loss_history_Epoches.get(loss_name, [])) > 0:
            plt.plot(epochs, loss_history_Epoches[loss_name], label=label, color=color, linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Time (Epoch)")
    plt.legend()
    plt.grid(True)

    plt.xticks(range(1, epochs_count + 1))
    plt.xlim([1, epochs_count])

    # Status-text for evaluating the performance
    Final_loss = loss_history_Epoches["combined"][-1] if len(loss_history_Epoches["combined"]) > 0 else 0
    if epochs_count > 0:
        if Final_loss < 0.01:
            status_text = f"Epoch {epochs_count}: Outstanding! Loss is exceptionally low ({Final_loss:.4f}). Great job!"
        elif Final_loss < 0.05:
            status_text = f"Epoch {epochs_count}: Excellent progress! Loss is impressively low ({Final_loss:.4f}). Keep it up!"
        elif Final_loss < 0.1:
            status_text = f"Epoch {epochs_count}: Good performance. Loss is low ({Final_loss:.4f}). Stay consistent!"
        elif Final_loss < 0.2:
            status_text = f"Epoch {epochs_count}: Decent. Loss is moderate ({Final_loss:.4f}). Fine-tuning might help!"
        elif Final_loss < 0.3:
            status_text = f"Epoch {epochs_count}: Loss is slightly high ({Final_loss:.4f}). Consider optimizing your model or data."
        else:
            status_text = f"Epoch {epochs_count}: Loss is high ({Final_loss:.4f}). Investigate potential issues with training or model."
    else:
        status_text = "No data available for evaluation."

    plt.text(0.5, 0.5, status_text, fontsize=12, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.7))

    # Save the plot
    base, ext = os.path.splitext(out_path)
    if ext.lower() not in [".png", ".jpg", ".jpeg", ".svg", ".pdf"]:
        out_path = base + ".png"

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    loss_logger.info("Generated plot_loss_curves_Training_script_epoches diagram.")










###TRAINING-BATCHES####
def plot_loss_curves_Training_script_Batches(loss_history_Batches, loss_logger,out_path="loss_curves_training_batches.png"):
    
    batch_count = len(loss_history_Batches["combined"])
    batch_range = range(1, batch_count + 1)

    loss_logger.info(f"Batch Count: {batch_count}, Combined Loss Entries: {len(loss_history_Batches['combined'])}")

    plt.figure(figsize=(14, 7))  

    # Plot L1-loss
    if len(loss_history_Batches["mask_loss"]) > 0:
        plt.plot(batch_range, loss_history_Batches["mask_loss"], label="mask_loss", color="blue", linewidth=1.5, alpha=0.8)

    # Plot Spectral/MSE-loss
    if len(loss_history_Batches["hybrid_loss"]) > 0:
        plt.plot(batch_range, loss_history_Batches["hybrid_loss"], label="hybrid_loss", color="green", linewidth=1.5, alpha=0.8)

    # Plot Combined-loss
    plt.plot(batch_range, loss_history_Batches["combined"], label="Combined-loss", color="purple", linewidth=2, alpha=0.9)

    # Adding labels and title
    plt.xlabel("Batch")
    plt.ylabel("Loss (log scale)")
    plt.title("Loss Over Time (Batch)")

    # Customize ticks and grid
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.yscale("log") 
    if batch_count > 20:
        plt.xticks(range(1, batch_count + 1, max(1, batch_count // 10)))
    plt.xlim([1, batch_count])

    # Adding legend
    plt.legend()

    # Generate status text
    if batch_count > 0:
        Final_loss = loss_history_Batches["combined"][-1]
        if Final_loss < 0.01:
            status_text = f"Batch {batch_count}: Exceptional performance! Loss is nearly zero ({Final_loss:.4f})."
        elif Final_loss < 0.05:
            status_text = f"Batch {batch_count}: Excellent progress! Minimal loss ({Final_loss:.4f})."
        elif Final_loss < 0.1:
            status_text = f"Batch {batch_count}: Great job! Loss is under control ({Final_loss:.4f})."
        elif Final_loss < 0.2:
            status_text = f"Batch {batch_count}: Good work, but there's room for improvement ({Final_loss:.4f})."
        elif Final_loss < 0.3:
            status_text = f"Batch {batch_count}: Moderate performance. Consider tuning the model or data ({Final_loss:.4f})."
        elif Final_loss < 0.5:
            status_text = f"Batch {batch_count}: Loss is slightly high. Try adjusting hyperparameters ({Final_loss:.4f})."
        else:
            status_text = f"Batch {batch_count}: High loss detected ({Final_loss:.4f}). Focus on debugging or re-evaluating the approach."
    else:
        status_text = "No data available for evaluation."

    # Add status text to the plot
    plt.text(0.5, 0.9, status_text, fontsize=12, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.7))

    # Save the plot
    base, ext = os.path.splitext(out_path)
    if ext.lower() not in [".png", ".jpg", ".jpeg", ".svg", ".pdf"]:
        out_path = base + ".png"

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    loss_logger.info("Generated plot_loss_curves_Training_script_Batches diagram.")















####FINE-TUNING#####

def plot_loss_curves_FineTuning_script_(loss_history_finetuning_epoches, out_path=os.path.join(root_dir,"Model_performance_logg/finetuning.png")):
    epochs_count = len(loss_history_finetuning_epoches["combined"])
    epochs = range(1, epochs_count + 1)
    plt.figure(figsize=(10,6))


    if len(loss_history_finetuning_epoches["l1"]) > 0:
        plt.plot(epochs, loss_history_finetuning_epoches["l1"], label="L1-loss", color="blue")


    if len(loss_history_finetuning_epoches["spectral"]) > 0:
        plt.plot(epochs, loss_history_finetuning_epoches["spectral"], label="spectral-loss", color="green")

    if len(loss_history_finetuning_epoches["perceptual"]) > 0:
        plt.plot(epochs, loss_history_finetuning_epoches["perceptual"], label="l1-loss", color="red")

    if len(loss_history_finetuning_epoches["mse"]) > 0:
        plt.plot(epochs, loss_history_finetuning_epoches["mse"], label="mse-loss", color="black")

    if len(loss_history_finetuning_epoches["multiscale"]) > 0:
        plt.plot(epochs, loss_history_finetuning_epoches["multiscale"], label="multiscale-loss", color="orange")



    plt.plot(epochs, loss_history_finetuning_epoches["combined"], label="combined-loss", color="purple")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Time (Batch)")
    plt.legend()


    Final_loss = loss_history_finetuning_epoches["combined"][-1] if len(loss_history_finetuning_epoches["combined"]) > 0 else 0
    if epochs_count  > 0:
        Final_loss = loss_history_finetuning_epoches["combined"][-1]
        if Final_loss < 0.01:
            status_text = f"Epoch {epochs}: Phenomenal ({Final_loss:.4f})"
        elif Final_loss < 0.05:
            status_text = f"Epoch {epochs}: Excellent (loss={Final_loss:.4f})"
        elif Final_loss < 0.1:
            status_text = f"Epoch {epochs}: Good (loss={Final_loss:.4f})"
        elif Final_loss < 0.2:
            status_text = f"Epoch {epochs}: Keep going (loss={Final_loss:.4f})"
        else:
            status_text = f"Epoch {epochs}: Higher loss ({Final_loss:.4f})"
    else:
        status_text = "No data"

    plt.text(0.5, 0.5, status_text, fontsize=12, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.7))


    base, ext = os.path.splitext(out_path)
    if ext.lower() not in [".png", ".jpg", ".jpeg", ".svg", ".pdf"]:
        out_path = base + ".png"

    plt.savefig(out_path)
    plt.close()
    print("Generated plot_loss_curves_FineTuning_script_ diagram.")



    




def visualize_and_save_waveforms(Fine_tune_logger,gt_waveform, pred_waveform, sample_idx, epoch, save_dir):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title(f"Epoch {epoch+1} - Ground Truth Sample {sample_idx+1}")
    librosa.display.waveshow(gt_waveform, sr=16000)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.subplot(1, 2, 2)
    plt.title(f"Epoch {epoch+1} - Predicted Sample {sample_idx+1}")
    librosa.display.waveshow(pred_waveform, sr=16000)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"epoch_{epoch+1}_sample_{sample_idx+1}.png")
    plt.savefig(save_path)
    plt.close()
    Fine_tune_logger.info(f"Saved waveform visualization at {save_path}")











def evaluate_metrics_from_spectrograms(Fine_tune_logger,ground_truth, predicted, loss_function, n_fft=1024, hop_length=512):
    Fine_tune_logger.debug("Evaluating metrics from spectrograms.")
    if predicted.size(1) != ground_truth.size(1):
        Fine_tune_logger.warning(
            f"Channel mismatch: predicted {predicted.size(1)}, ground truth {ground_truth.size(1)}. Adjusting predicted channels."
        )
        predicted = predicted[:, :ground_truth.size(1), :, :]
    if predicted.size() != ground_truth.size():
        raise ValueError(f"Shape mismatch in evaluation! Predicted: {predicted.size()}, Ground Truth: {ground_truth.size()}")
    gt_waveforms = loss_function.spectrogram_to_waveform(ground_truth, n_fft, hop_length)
    pred_waveforms = loss_function.spectrogram_to_waveform(predicted, n_fft, hop_length)
    sdr_list, sir_list, sar_list = [], [], []
    for idx, (gt, pred) in enumerate(zip(gt_waveforms, pred_waveforms)):
        if np.allclose(gt, 0):
            Fine_tune_logger.info(f"Skipping evaluation for a silent reference in sample {idx+1}.")
            continue
        min_len = min(len(gt), len(pred))
        gt = gt[:min_len]
        pred = pred[:min_len]
        try:
            sdr, sir, sar, _ = bss_eval_sources(gt[np.newaxis, :], pred[np.newaxis, :])
            sdr_list.append(sdr[0])
            sir_list.append(sir[0])
            sar_list.append(sar[0])
            Fine_tune_logger.debug(f"Metrics for sample {idx+1} - SDR: {sdr[0]:.4f}, SIR: {sir[0]:.4f}, SAR: {sar[0]:.4f}")
        except Exception as e:
            Fine_tune_logger.error(f"Error evaluating metrics for sample {idx+1}: {e}")
            continue
    Fine_tune_logger.debug("Metrics evaluation completed.")
    return sdr_list, sir_list, sar_list












