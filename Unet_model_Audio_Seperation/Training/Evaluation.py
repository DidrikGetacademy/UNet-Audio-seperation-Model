import os
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from torch import autocast
from Training.Externals.Logger import setup_logger
from mir_eval.separation import bss_eval_sources
from Model_Architecture.model import UNet
from Training.Externals.Dataloader import create_dataloaders
from Training.Externals.Functions import Return_root_dir
import deepspeed

root_dir = Return_root_dir()
eval_log_path = os.path.join(root_dir, "Model_Performance_logg/log/Evaluation_logg.txt")
Evaluation_logger = setup_logger('Evaluation', eval_log_path)
checkpoint_path = os.path.join(root_dir, "Model_Weights/Best_model.pth") 

# Load DeepSpeed config
with open(os.path.join(root_dir, "DeepSeed_Configuration/ds_config.json"), "r") as f:
    ds_config = json.load(f)

# Create data loaders (use validation set or test set as needed)
train_loader, val_loader_phase = create_dataloaders(
    batch_size=ds_config["train_micro_batch_size_per_gpu"],
    num_workers=0,
    sampling_rate=44100,
    max_length_seconds=10,
    max_files_train=50,
    max_files_val=20,
)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Evaluation_logger.info(f"Using device: {device}")

# Initialize the model
model = UNet(in_channels=1, out_channels=1).to(device)
model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    config_params=ds_config
)

# Load the best model checkpoint
def load_model_for_evaluation(checkpoint_path, model_engine, device):
    try:
        model_engine.load_checkpoint(checkpoint_path)
        Evaluation_logger.info(f"Model loaded from {checkpoint_path}")
    except Exception as e:
        Evaluation_logger.error(f"Error loading model checkpoint: {e}")
        raise

load_model_for_evaluation(checkpoint_path, model_engine, device)

# Evaluation Metrics
def evaluate_metrics_from_spectrograms(ground_truth, predicted, n_fft=1024, hop_length=512):
    """
    Compute SDR, SIR, and SAR from spectrograms (using mir_eval).
    """
    gt_waveforms = model_engine.module.spectrogram_to_waveform(ground_truth, n_fft, hop_length)
    pred_waveforms = model_engine.module.spectrogram_to_waveform(predicted, n_fft, hop_length)
    sdr_list, sir_list, sar_list = [], [], []
    
    for gt, pred in zip(gt_waveforms, pred_waveforms):
        sdr, sir, sar, _ = bss_eval_sources(gt[np.newaxis, :], pred[np.newaxis, :])
        sdr_list.append(sdr[0])
        sir_list.append(sir[0])
        sar_list.append(sar[0])

    return sdr_list, sir_list, sar_list

# Inference & Evaluation
def run_evaluation(model_engine, val_loader, device, save_visualizations_every_n_batches=5):
    model_engine.eval()
    sdr_list, sir_list, sar_list = [], [], []
    running_loss = 0.0

    with torch.no_grad(), autocast(device_type='cuda', enabled=(device.type == 'cuda')):
        for batch_idx, (inputs, targets) in enumerate(val_loader, start=1):
            inputs, targets = inputs.to(device), targets.to(device)

            predicted_mask, outputs = model_engine(inputs)
            combined_loss, _, _ = model_engine.module.loss_function(predicted_mask, inputs, targets, outputs)
            running_loss += combined_loss.item()

            # Evaluate metrics
            batch_sdr, batch_sir, batch_sar = evaluate_metrics_from_spectrograms(targets, outputs)
            sdr_list.extend(batch_sdr)
            sir_list.extend(batch_sir)
            sar_list.extend(batch_sar)

            # Visualize results for selected batches (every 'n' batches)
            if batch_idx % save_visualizations_every_n_batches == 0:
                visualize_results(inputs, outputs, targets, batch_idx)

    avg_loss = running_loss / len(val_loader)
    avg_sdr = np.mean(sdr_list)
    avg_sir = np.mean(sir_list)
    avg_sar = np.mean(sar_list)

    Evaluation_logger.info(f"Evaluation Loss: {avg_loss:.6f}")
    Evaluation_logger.info(f"Average SDR: {avg_sdr:.4f}, Average SIR: {avg_sir:.4f}, Average SAR: {avg_sar:.4f}")
    
    return avg_loss, avg_sdr, avg_sir, avg_sar

# Visualization function to save waveform comparison
def visualize_results(inputs, outputs, targets, batch_idx):
    """
    Visualize and save the waveform comparison between GT and predicted output for the selected batch.
    """
    gt_waveform = targets[0].cpu().numpy().flatten()
    pred_waveform = outputs[0].cpu().numpy().flatten()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    librosa.display.waveshow(gt_waveform, sr=44100)
    plt.title(f"Ground Truth - Batch {batch_idx}")
    plt.subplot(1, 2, 2)
    librosa.display.waveshow(pred_waveform, sr=44100)
    plt.title(f"Predicted Output - Batch {batch_idx}")
    plt.tight_layout()

    # Save the visualizations
    save_path = os.path.join(root_dir, "visualizations", f"waveform_comparison_batch_{batch_idx}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    Evaluation_logger.info(f"Saved waveform comparison to {save_path}")

if __name__ == "__main__":
    avg_loss, avg_sdr, avg_sir, avg_sar = run_evaluation(model_engine, val_loader_phase, device)
    Evaluation_logger.info(f"Final Evaluation - Loss: {avg_loss:.6f}, SDR: {avg_sdr:.4f}, SIR: {avg_sir:.4f}, SAR: {avg_sar:.4f}")
