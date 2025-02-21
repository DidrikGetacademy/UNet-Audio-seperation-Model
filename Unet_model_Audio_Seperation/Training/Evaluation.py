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
from Training.Externals.Dataloader import create_dataloader_EVALUATION
from Training.Externals.Functions import Return_root_dir
from Training.Externals.Loss_Class_Functions import Combinedloss 
import json 
from Datasets.Scripts.Dataset_utils import spectrogram_to_waveform
root_dir = Return_root_dir()
eval_log_path = os.path.join(root_dir, "Model_Performance_logg/log/Evaluation_logg.txt")
Evaluation_logger = setup_logger('Evaluation', eval_log_path)
checkpoint_path = os.path.join(root_dir, "Model_Weights/Best_model.pth") 

# Load DeepSpeed config
with open(os.path.join(root_dir, "DeepSeed_Configuration/ds_config_Training.json"), "r") as f:
    ds_config = json.load(f)

# Create data loaders (use validation set or test set as needed)
Evaluation_Loader = create_dataloader_EVALUATION(
    batch_size=ds_config["train_micro_batch_size_per_gpu"],
    num_workers=6,
    sampling_rate=44100,
    max_length_seconds=10,
    max_files_val=None,
)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize the model
model = UNet(in_channels=1, out_channels=1).to(device)


# Load the best model checkpoint
def load_model_for_evaluation(checkpoint_path, model_engine, device=device):
    try:
        model_engine.load_checkpoint(checkpoint_path)
        Evaluation_logger.info(f"Model loaded from {checkpoint_path}")
    except Exception as e:
        Evaluation_logger.error(f"Error loading model checkpoint: {e}")
        raise


# Evaluation Metrics
def evaluate_metrics_from_spectrograms(ground_truth, predicted, n_fft=1024, hop_length=512):
    gt_waveforms = spectrogram_to_waveform(ground_truth, n_fft, hop_length)
    pred_waveforms = spectrogram_to_waveform(predicted, n_fft, hop_length)
    sdr_list, sir_list, sar_list = [], [], []
    
    for gt, pred in zip(gt_waveforms, pred_waveforms):
        sdr, sir, sar, _ = bss_eval_sources(gt[np.newaxis, :], pred[np.newaxis, :])
        sdr_list.append(sdr[0])
        sir_list.append(sir[0])
        sar_list.append(sar[0])

    return sdr_list, sir_list, sar_list
criterion = Combinedloss()
# Inference & Evaluation
def run_evaluation(model_engine, device, save_visualizations_every_n_batches=5):
    load_model_for_evaluation(checkpoint_path, model_engine=model_engine, device=device)
    model_engine.eval()
    sdr_list, sir_list, sar_list = [], [], []
    running_loss = 0.0

    with torch.no_grad(), autocast(device_type='cuda', enabled=(device.type == 'cuda')):
        for batch_idx, (inputs, targets) in enumerate(Evaluation_Loader, start=1):
            inputs, targets = inputs.to(device), targets.to(device)
            predicted_mask, outputs = model_engine(inputs)
            Evaluation_logger.info(f"Predicted shape: {predicted_mask.shape},\n Target shape: {targets.shape}\n inputs shape: {inputs.shape}\n outputs shape: {outputs.shape}\n")

            combined_loss, mask_loss, hybrid_loss_val, l1_loss_val, stft_loss_val, sdr_loss  = criterion(predicted_mask, inputs, targets)
            Evaluation_logger.info(f"Combined Loss: {combined_loss:.6f},\n Mask Loss: {mask_loss:.6f},\n Hybrid Loss: {hybrid_loss_val:.6f},\n L1 Loss: {l1_loss_val:.6f},\n STFT Loss: {stft_loss_val:.6f},\n SDR Loss: {sdr_loss:.6f}")
            running_loss += combined_loss.item()
            Evaluation_logger.info(f"Running Loss: {running_loss:.6f}")

        
            batch_sdr, batch_sir, batch_sar = evaluate_metrics_from_spectrograms(targets, outputs)
            sdr_list.extend(batch_sdr)
            sir_list.extend(batch_sir)
            sar_list.extend(batch_sar)


            if batch_idx % save_visualizations_every_n_batches == 0:
                visualize_results(inputs, outputs, targets, batch_idx)

    avg_loss = running_loss / len(Evaluation_Loader)
    avg_sdr = np.mean(sdr_list)
    avg_sir = np.mean(sir_list)
    avg_sar = np.mean(sar_list)

    Evaluation_logger.info(f"Evaluation Loss: {avg_loss:.6f}[BØR REDUSERES OVER TID]")
    Evaluation_logger.info(f"[Average SDR]->[SDR (Signal-to-Distortion Ratio): Måler hvor mye uønsket støy modellen introduserer. Høyere er bedre.]: avg_sdr={avg_sdr:.4f}\n" 
                           f"[Average SIR]-> [SIR (Signal-to-Interference Ratio): Måler hvor godt modellen separerer vokaler fra instrumenter. Høyere er bedre.]: avg_sir={avg_sir:.4f}\n"
                           f"[Average SAR]-> [SAR (Signal-to-Artifacts Ratio): Måler hvor mye kunstige feil modellen introduserer. Også høyere er bedre.]: avg:sar={avg_sar:.4f}\n"
                           )
    
    Evaluation_logger.info(f"Final Evaluation - Loss: {avg_loss:.6f}, SDR: {avg_sdr:.4f}, SIR: {avg_sir:.4f}, SAR: {avg_sar:.4f}")
    return avg_loss, avg_sdr, avg_sir, avg_sar


def visualize_results(inputs, outputs, targets, batch_idx):

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


    save_path = os.path.join(root_dir, "visualizations", f"waveform_comparison_batch_{batch_idx}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    Evaluation_logger.info(f"Saved waveform comparison to {save_path}")
