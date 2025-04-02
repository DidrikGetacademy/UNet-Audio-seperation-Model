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

# Load DeepSpeed config
with open(os.path.join(root_dir, "DeepSeed_Configuration/ds_config_Training.json"), "r") as f:
    ds_config = json.load(f)

# Initialize dataloader
Evaluation_Loader = create_dataloader_EVALUATION(
    batch_size=ds_config["train_micro_batch_size_per_gpu"],
    num_workers=6,
    sampling_rate=44100,
    max_length_seconds=11,
    max_files_val=30,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = Combinedloss()
model = UNet(in_channels=1, out_channels=1).to(device)
def load_model_for_evaluation(checkpoint_dir, model_engine):
    try:
     
        load_path, client_state = model_engine.load_checkpoint(
            checkpoint_dir,
            tag="best_model",
            load_module_only=False
        )
        if load_path is None:
            raise ValueError(f"No checkpoint found in {checkpoint_dir}")
        Evaluation_logger.info(f"Loaded checkpoint from {load_path}")
    except Exception as e:
        Evaluation_logger.error(f"Load failed: {str(e)}")
        raise

def evaluate_metrics(predicted_spec, ground_truth_waveform, n_fft=1024, hop_length=512):
    """Calculate separation metrics with sanity checks"""

    pred_waveform = spectrogram_to_waveform(predicted_spec, n_fft, hop_length)
    
    # Validate input lengths
    gt_np = ground_truth_waveform.cpu().numpy()
    pred_np = pred_waveform.cpu().numpy()
    
    if len(gt_np) == 0 or len(pred_np) == 0:
        Evaluation_logger.error("Empty waveforms in metric calculation")
        return [], [], []
    
    # Calculate metrics with safety checks
    sdr_list, sir_list, sar_list = [], [], []
    for gt, pred in zip(gt_np, pred_np):
        min_len = min(len(gt), len(pred))
        gt_energy = np.sum(gt[:min_len] ** 2)
        if gt_energy < 1e-8:  # Near-zero energy
            Evaluation_logger.warning("Skipping silent ground truth segment.")
            continue
        if min_len < 1024:  # Minimum length for meaningful evaluation
            Evaluation_logger.warning(f"Short audio segment ({min_len} samples)")
            continue
            
        try:
            sdr, sir, sar, _ = bss_eval_sources(
                gt[:min_len][np.newaxis, :], 
                pred[:min_len][np.newaxis, :]
            )
            sdr_list.append(sdr[0])
            sir_list.append(sir[0])
            sar_list.append(sar[0])
        except ValueError as e:
            Evaluation_logger.error(f"Metric calculation failed: {str(e)}")
            
    return sdr_list, sir_list, sar_list

def run_evaluation(model_engine, device, checkpoint_path, save_visualizations_every_n_batches=5):
    """Main evaluation pipeline with enhanced logging"""
    load_model_for_evaluation(checkpoint_path, model_engine)
    model_engine.eval()
    
    n_fft = 1024
    hop_length = 512

    sdr_list, sir_list, sar_list = [], [], []
    running_loss = 0.0
    silent_predictions = 0

    with torch.no_grad(), autocast(device_type='cuda', enabled=(device.type == 'cuda')):
        for batch_idx, (inputs, targets) in enumerate(Evaluation_Loader, 1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            target_waveforms = spectrogram_to_waveform(targets, n_fft, hop_length)

            # Forward pass
            predicted_mask, outputs = model_engine(inputs)
            loss = criterion(predicted_mask, inputs, targets)
            running_loss += loss.item()

            # Metric calculation
            batch_sdr, batch_sir, batch_sar = evaluate_metrics(
                outputs, target_waveforms, n_fft, hop_length
            )
            
            # Silent prediction check
            pred_waveforms = spectrogram_to_waveform(outputs, n_fft, hop_length)
            silent_predictions += sum(torch.max(wf).item() < 0.01 for wf in pred_waveforms)

            # Store metrics
            sdr_list.extend(batch_sdr)
            sir_list.extend(batch_sir)
            sar_list.extend(batch_sar)

            # Visualization
            if batch_idx % save_visualizations_every_n_batches == 0:
                visualize_results(inputs, outputs, target_waveforms, batch_idx)

    # Calculate averages
    avg_loss = running_loss / len(Evaluation_Loader)
    avg_sdr = np.mean(sdr_list) if sdr_list else 0
    avg_sir = np.mean(sir_list) if sir_list else 0
    avg_sar = np.mean(sar_list) if sar_list else 0

    # Final logging
    Evaluation_logger.info("\n=== Evaluation Summary ===")
    Evaluation_logger.info(f"Processed {len(sdr_list)} valid samples")
    Evaluation_logger.info(f"Average Loss: {avg_loss:.4f}")
    Evaluation_logger.info(f"SDR: {avg_sdr:.2f} ± {np.std(sdr_list):.2f} dB")
    Evaluation_logger.info(f"SIR: {avg_sir:.2f} ± {np.std(sir_list):.2f} dB")
    Evaluation_logger.info(f"SAR: {avg_sar:.2f} ± {np.std(sar_list):.2f} dB")
    
    # Performance benchmarks
    Evaluation_logger.info("\n=== Performance Benchmarks ===")
    Evaluation_logger.info(
        "Metric\t\tExpected\tGood\t\tSOTA\n"
        "------------------------------------------------\n"
        "SDR\t\t3-7 dB\t\t>5 dB\t\t7-9 dB\n"
        "SIR\t\t8-15 dB\t\t>10 dB\t\t12-16 dB\n"
        "SAR\t\t5-10 dB\t\t>7 dB\t\t9-12 dB"
    )
    
    # Training stage comparison
    Evaluation_logger.info("\n=== Performance Context ===")
    Evaluation_logger.info(
        "Stage\t\t\tSDR\t\tSIR\t\tSAR\n"
        "------------------------------------------------\n"
        "Initial (1-2 epochs)\t2-4 dB\t\t5-8 dB\t\t3-5 dB\n"
        "Converged\t\t5-7 dB\t\t10-14 dB\t7-9 dB\n"
        "SOTA Reference\t\t7-9 dB\t\t12-16 dB\t9-12 dB"
    )
    
    # Anomaly detection
    Evaluation_logger.info("\n=== Quality Checks ===")
    if silent_predictions > 0:
        Evaluation_logger.warning(f"Silent predictions detected: {silent_predictions}")
    if any(sdr < -5 for sdr in sdr_list):
        Evaluation_logger.error("Invalid SDR values (<-5 dB) detected!")
    if avg_sdr > 10 or avg_sir > 20:
        Evaluation_logger.warning("Suspiciously high metrics - verify ground truth alignment")

    return avg_loss, avg_sdr, avg_sir, avg_sar

def visualize_results(inputs, outputs, target_waveforms, batch_idx):
    """Enhanced visualization with waveforms and spectrograms"""
    try:
        # Convert tensors to numpy
        gt_waveform = target_waveforms[0].cpu().numpy().flatten()
        pred_waveform = spectrogram_to_waveform(outputs[0]).cpu().numpy().flatten()

        plt.figure(figsize=(18, 12))
        
        # Waveform plot
        plt.subplot(2, 2, 1)
        librosa.display.waveshow(gt_waveform, sr=44100)
        plt.title(f"Ground Truth Waveform (Batch {batch_idx})")
        
        plt.subplot(2, 2, 2)
        librosa.display.waveshow(pred_waveform, sr=44100)
        plt.title(f"Predicted Waveform (Batch {batch_idx})")
        
        # Spectrogram plot
        plt.subplot(2, 2, 3)
        S = librosa.stft(gt_waveform, n_fft=1024, hop_length=512)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(S), ref=np.max),
                               sr=44100, hop_length=512, x_axis='time', y_axis='log')
        plt.title("Ground Truth Spectrogram")
        plt.colorbar(format='%+2.0f dB')

        plt.subplot(2, 2, 4)
        S_pred = librosa.stft(pred_waveform, n_fft=1024, hop_length=512)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(S_pred), ref=np.max),
                               sr=44100, hop_length=512, x_axis='time', y_axis='log')
        plt.title("Predicted Spectrogram")
        plt.colorbar(format='%+2.0f dB')

        # Save and close
        save_path = os.path.join(root_dir, "visualizations", f"eval_batch_{batch_idx}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        Evaluation_logger.info(f"Saved visualization: {save_path}")
        
    except Exception as e:
        Evaluation_logger.error(f"Visualization failed for batch {batch_idx}: {str(e)}")

if __name__ == "__main__":
    import deepspeed
    model = UNet(in_channels=1, out_channels=1).to(device)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config_params=ds_config
    )
    checkpoint_path = os.path.join(root_dir, "Model_Weights/Pre_trained", "deepspeed_checkpoint")
    run_evaluation(model_engine,device,checkpoint_path)