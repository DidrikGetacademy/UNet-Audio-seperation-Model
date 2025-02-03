import os
import sys
import torch
import deepspeed
import librosa
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import traceback
import matplotlib.pyplot as plt
import librosa.display
from torch import autocast
from mir_eval.separation import bss_eval_sources
import json 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Training.Externals.utils import Return_root_dir
from Training.Externals.Logger import setup_logger
from Training.Externals.Dataloader import create_dataloaders
from Training.Externals.Loss_Diagram_Values import plot_loss_curves_FineTuning_script_
from Training.Externals.Functions import load_model_path_func
root_dir = Return_root_dir() 
fine_tuned_model_path = os.path.join(root_dir, "Model_Weights/Fine_Tuned/Model.pth")
train_log_path = os.path.join(root_dir, "Model_Performance_logg/log/Fine_tune.txt")
Fine_tune_logger = setup_logger('Fine-Tuning', train_log_path)
with open(os.path.join(root_dir, "DeepSeed_Configuration/ds_config.json"), "r") as f:
    ds_config = json.load(f)



world_size = torch.cuda.device_count()  
train_batch_size = ds_config["train_batch_size"]
train_micro_batch_size_per_gpu = train_batch_size // world_size
gradient_accumulation_steps = train_batch_size // (train_micro_batch_size_per_gpu * world_size)
assert train_micro_batch_size_per_gpu * gradient_accumulation_steps * world_size == train_batch_size, \
    f"Inconsistent batch sizes: {train_micro_batch_size_per_gpu} * {gradient_accumulation_steps} * {world_size} != {train_batch_size}"

# Update DeepSpeed config dynamically
ds_config["train_micro_batch_size_per_gpu"] = train_micro_batch_size_per_gpu
ds_config["gradient_accumulation_steps"] = gradient_accumulation_steps
ds_config["world_size"] = world_size 

torch.backends.cudnn.benchmark = True 
torch.set_num_threads(8)

# Global loss history
loss_history_finetuning_epoches = {
    "l1": [],
    "mse": [],
    "spectral": [],
    "perceptual": [],
    "multiscale": [],
    "combined": [],
}

def Append_loss_values(loss_history, epoch_losses, epoch):
    """
    Append averaged loss values over the epoch.
    """
    for key in loss_history.keys():
        loss_history[key].append(epoch_losses[key] / epoch_losses["count"])
    
    log_message = (
        f"[Fine-Tune] Epoch {epoch+1} | "
        f"L1 Loss={loss_history['l1'][-1]:.6f}, MSE Loss={loss_history['mse'][-1]:.6f}, "
        f"Spectral Loss={loss_history['spectral'][-1]:.6f}, Perceptual Loss={loss_history['perceptual'][-1]:.6f}, "
        f"Multi-Scale Loss={loss_history['multiscale'][-1]:.6f}, Combined Loss={loss_history['combined'][-1]:.6f}"
    )
    Fine_tune_logger.info(log_message)

class CombinedLoss(nn.Module):
    def __init__(self, device):
        super(CombinedLoss, self).__init__()
        self.device = device
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        Fine_tune_logger.info("Loading VGGish model for perceptual loss...")
        # Load VGGish from Torch Hub. If available, check documentation for batching.
        self.audio_model = torch.hub.load('harritaylor/torchvggish', 'vggish', trust_repo=True).to(device)
        self.audio_model.eval()
        Fine_tune_logger.info("VGGish model loaded and set to evaluation mode.")

    def preprocess_for_vggish(self, waveforms, fs=16000):
        """
        Preprocess a batch of waveforms for VGGish.
        This may include normalization and conversion to log mel spectrograms.
        Adjust this function to match the VGGish expectations.
        """
        # Ensure waveforms are torch tensors with shape (B, L)
        waveforms = torch.stack([torch.from_numpy(w).float() for w in waveforms], dim=0).to(self.device)
        # Example normalization: ensure the waveform amplitude is in [-1, 1]
        waveforms = waveforms / (waveforms.abs().max(dim=1, keepdim=True)[0] + 1e-9)
        return waveforms

    def spectrogram_to_waveform(self, spectrogram, n_fft=1024, hop_length=512):
        """
        Converts a batch (or single) spectrogram tensor into a list of waveforms.
        """
        Fine_tune_logger.debug("Converting spectrogram(s) to waveform(s) using Griffin-Lim.")
        spectrogram_np = spectrogram.detach().cpu().numpy()
        # Support both batched and single-sample spectrograms
        if spectrogram_np.ndim == 3:
            # Shape (B, F, T)
            spectrogram_np = spectrogram_np[:, None, ...]  # Add a channel dimension for consistency
        waveforms = []
        for i in range(spectrogram_np.shape[0]):
            # Assume first channel contains the magnitude
            mag = spectrogram_np[i, 0]
            waveform = librosa.griffinlim(mag, n_iter=32, hop_length=hop_length, win_length=n_fft)
            waveforms.append(waveform)
            Fine_tune_logger.debug(f"Sample {i+1}: Waveform length after Griffin-Lim: {len(waveform)}")
        Fine_tune_logger.debug("Spectrogram to waveform conversion completed.")
        return waveforms

    def mask_loss(self, predicted_mask, mixture, target):
        Fine_tune_logger.debug("Calculating mask loss.")
        if predicted_mask.size() != mixture.size():
            raise ValueError(f"Shape mismatch: predicted_mask {predicted_mask.size()} and mixture {mixture.size()}!")
        predicted_vocals = predicted_mask * mixture
        if predicted_vocals.size() != target.size():
            raise ValueError(f"Shape mismatch: predicted_vocals {predicted_vocals.size()} and target {target.size()}!")
        l1 = self.l1_loss(predicted_vocals, target)
        stft = self.mse_loss(torch.log1p(predicted_vocals), torch.log1p(target))
        total_mask_loss = 0.5 * l1 + 0.5 * stft
        return total_mask_loss

    def perceptual_loss(self, output, target):
        Fine_tune_logger.debug("Calculating perceptual loss.")
        # Convert spectrograms to waveforms for the whole batch
        output_waveforms = self.spectrogram_to_waveform(output)
        target_waveforms = self.spectrogram_to_waveform(target)
        batch_size = len(output_waveforms)
        Fine_tune_logger.debug(f"Batch size for perceptual loss: {batch_size}")

        orig_sr, new_sr = 44100, 16000
        max_length = 176400  # 11 seconds max; adjust as needed

        processed_output = []
        processed_target = []
        for i in range(batch_size):
            out_np = output_waveforms[i]
            tgt_np = target_waveforms[i]
            # Resample using librosa
            out_16k = librosa.resample(out_np, orig_sr=orig_sr, target_sr=new_sr)
            tgt_16k = librosa.resample(tgt_np, orig_sr=orig_sr, target_sr=new_sr)
            # Pad or truncate to max_length
            if len(out_16k) < max_length:
                out_16k = np.pad(out_16k, (0, max_length - len(out_16k)))
            else:
                out_16k = out_16k[:max_length]
            if len(tgt_16k) < max_length:
                tgt_16k = np.pad(tgt_16k, (0, max_length - len(tgt_16k)))
            else:
                tgt_16k = tgt_16k[:max_length]
            processed_output.append(out_16k)
            processed_target.append(tgt_16k)

        # Preprocess the batch for VGGish
        processed_output = self.preprocess_for_vggish(processed_output, fs=new_sr)
        processed_target = self.preprocess_for_vggish(processed_target, fs=new_sr)

        # Attempt to use batched processing for VGGish.
        with torch.no_grad():
            try:
                output_features = self.audio_model(processed_output, fs=new_sr)
                target_features = self.audio_model(processed_target, fs=new_sr)
                Fine_tune_logger.debug("Batched VGGish feature extraction succeeded.")
            except Exception as e:
                # Fallback to per-sample extraction if batching fails.
                Fine_tune_logger.error(f"Batched VGGish feature extraction failed: {e}. Falling back to per-sample extraction.")
                output_features_list, target_features_list = [], []
                for i in range(batch_size):
                    try:
                        out_feat = self.audio_model(processed_output[i].unsqueeze(0), fs=new_sr)
                        tgt_feat = self.audio_model(processed_target[i].unsqueeze(0), fs=new_sr)
                        output_features_list.append(out_feat)
                        target_features_list.append(tgt_feat)
                    except Exception as e:
                        Fine_tune_logger.error(f"Error extracting VGGish features for sample {i+1}: {e}")
                        continue
                if not output_features_list or not target_features_list:
                    Fine_tune_logger.warning("No features extracted for perceptual loss. Returning zero loss.")
                    return torch.tensor(0.0, device=self.device)
                output_features = torch.cat(output_features_list, dim=0)
                target_features = torch.cat(target_features_list, dim=0)

        loss_value = self.mse_loss(output_features, target_features)
        Fine_tune_logger.debug(f"Perceptual Loss value: {loss_value.item():.6f}")
        return loss_value

    def normalize_waveform(self, waveform):
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            normalized = waveform / max_val
            if torch.isnan(normalized).any() or torch.isinf(normalized).any():
                Fine_tune_logger.warning("NaN or Inf detected after normalization; returning zeros.")
                return torch.zeros_like(waveform)
            return normalized
        else:
            return torch.zeros_like(waveform)

    def spectral_loss(self, output, target, n_fft=1024, hop_length=512):
        Fine_tune_logger.debug("Calculating spectral loss.")
        output = self.normalize_waveform(output)
        target = self.normalize_waveform(target)

        # Ensure the shape is (B, F, T) or (B, 1, F, T)
        if output.ndim == 4:
            output = output.squeeze(1)
            target = target.squeeze(1)
        elif output.ndim != 3:
            raise ValueError(f"Unexpected tensor dimensions for spectral loss. Output shape: {output.shape}, Target shape: {target.shape}")

        output_log = torch.log1p(torch.abs(output))
        target_log = torch.log1p(torch.abs(target))
        loss = self.mse_loss(output_log, target_log)
        Fine_tune_logger.debug(f"Spectral loss calculated: {loss.item():.6f}")
        return loss

    def multi_scale_loss(self, output, target, scales=[1, 2, 4]):
        Fine_tune_logger.debug("Calculating multi-scale loss.")
        total_loss = 0.0
        for scale in scales:
            scaled_output = F.avg_pool2d(output, kernel_size=scale)
            scaled_target = F.avg_pool2d(target, kernel_size=scale)
            loss = self.mse_loss(scaled_output, scaled_target)
            total_loss += loss
            Fine_tune_logger.debug(f"Scale {scale}: MSE loss={loss.item():.6f}")
        Fine_tune_logger.debug(f"Total multi-scale loss: {total_loss.item():.6f}")
        return total_loss

    def forward(self, predicted_mask, mixture, target, outputs):
        Fine_tune_logger.debug("Forward pass for combined loss calculation.")
        mask_loss_val = self.mask_loss(predicted_mask, mixture, target)
        l1_val = self.l1_loss(outputs, target)
        mse_val = self.mse_loss(outputs, target)
        spectral_val = self.spectral_loss(outputs, target)
        perceptual_val = self.perceptual_loss(outputs, target)
        multi_scale_val = self.multi_scale_loss(outputs, target)

        total_loss = (
            0.3 * l1_val +
            0.3 * mse_val +
            0.1 * spectral_val +
            0.2 * perceptual_val +
            0.1 * multi_scale_val
        )
        Fine_tune_logger.debug(f"Total loss (weighted sum): {total_loss.item():.6f}")

        combined_loss = 0.5 * mask_loss_val + 0.5 * total_loss
        Fine_tune_logger.debug(f"Combined loss (mask + total): {combined_loss.item():.6f}")
        return combined_loss, mask_loss_val, total_loss

def visualize_and_save_waveforms(gt_waveform, pred_waveform, sample_idx, epoch, save_dir):
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

def freeze_encoder(model):
    """
    Freeze the encoder parameters.
    This function now handles both cases: an encoder that is iterable (e.g., a list of layers)
    or a single module.
    """
    Fine_tune_logger.debug("Freezing encoder layers.")
    try:
        # Support for wrapped models (e.g., DeepSpeed or DataParallel)
        module = model.module if hasattr(model, 'module') else model
        encoder = getattr(module, 'encoder', None)
        if encoder is None:
            raise AttributeError("Model does not have an 'encoder' attribute.")
        # If encoder is iterable, freeze each layer; otherwise, freeze the module directly.
        if hasattr(encoder, '__iter__'):
            for layer in encoder:
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            for param in encoder.parameters():
                param.requires_grad = False
        encoder.eval()  
        Fine_tune_logger.info("Encoder layers frozen for fine-tuning.")
    except AttributeError as e:
        Fine_tune_logger.error(f"Error freezing encoder layers: {e}")
        raise

def resample_audio(waveform, orig_sr, target_sr=16000):
    Fine_tune_logger.debug("Resampling audio waveform.")
    if np.allclose(waveform, 0):
        Fine_tune_logger.warning("Silent audio detected during resampling.")
        return np.zeros(target_sr, dtype=np.float32)
    resampled = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
    max_val = np.max(np.abs(resampled))
    if max_val > 0:
        resampled = resampled / max_val 
        Fine_tune_logger.debug("Resampled waveform normalized.")
    else:
        Fine_tune_logger.warning("Max value after resampling is zero. Returning zeros.")
        resampled = np.zeros_like(resampled)
    return resampled

def evaluate_metrics_from_spectrograms(ground_truth, predicted, loss_function, n_fft=1024, hop_length=512):
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

# Create data loaders
train_loader, val_loader_phase = create_dataloaders(
    batch_size=ds_config["train_micro_batch_size_per_gpu"],
    num_workers=0,
    sampling_rate=44100,
    max_length_seconds=10,
    max_files_train=50,
    max_files_val=20,
)
Fine_tune_logger.info("Data loaders created successfully.")

def fine_tune_model(fine_tuned_model_path, Fine_tuned_training_loader, Finetuned_validation_loader, ds_config, fine_tune_epochs=10, pretrained_model_path=None):
    from Model_Architecture.model import UNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Fine_tune_logger.info(f"Fine-tuning --> Using device: {device}")
    load_model_path = os.path.join(root_dir, "Model_Weights/CheckPoints/checkpoint_epochsss_25")
    Fine_tune_logger.info("Initializing model...")
    model = UNet(in_channels=1, out_channels=1).to(device)

    # DeepSpeed Initialization
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters()
    )
    Fine_tune_logger.debug("DeepSpeed model_engine initialized.")
    
    if pretrained_model_path is None:
        load_model_path_func(load_model_path, model_engine, model, device)
    else:
        load_model_path_func(pretrained_model_path, model_engine, model, device)
    
    freeze_encoder(model_engine)
    loss_function = CombinedLoss(device)
    visualization_dir = os.path.join(os.path.dirname(fine_tuned_model_path), "visualizations")
    os.makedirs(visualization_dir, exist_ok=True)
    Fine_tune_logger.info(f"Visualization directory set to: {visualization_dir}")

    try:
        for epoch in range(fine_tune_epochs):
            Fine_tune_logger.info(f"Starting Epoch {epoch+1}/{fine_tune_epochs}")
            print(f"Starting Epoch {epoch+1}/{fine_tune_epochs}")
            model_engine.train()
            running_loss = 0.0
            # For averaging loss values across the epoch:
            epoch_losses = {"l1": 0.0, "mse": 0.0, "spectral": 0.0, "perceptual": 0.0, "multiscale": 0.0, "combined": 0.0, "count": 0}

            for batch_idx, (inputs, targets) in enumerate(Fine_tuned_training_loader, start=1):
                if inputs is None or targets is None:
                    Fine_tune_logger.warning(f"Skipping training batch {batch_idx} due to None data.")
                    continue
                Fine_tune_logger.debug(f"Training Batch {batch_idx}: Inputs shape={inputs.shape}, Targets shape={targets.shape}")
                inputs, targets = inputs.to(device), targets.to(device)
                model_engine.zero_grad()
                with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                    predicted_mask, outputs = model_engine(inputs)
                    # Slicing to keep one channel only if that is intended:
                    predicted_mask = predicted_mask[:, :1, :, :]
                    outputs = outputs[:, :1, :, :]
                    combined_loss, mask_loss, total_loss = loss_function(predicted_mask, inputs, targets, outputs)
                model_engine.backward(combined_loss)
                model_engine.step()

                running_loss += combined_loss.item()
                # Accumulate losses for epoch-level averaging
                epoch_losses["l1"] += loss_function.l1_loss(outputs, targets).item()
                epoch_losses["mse"] += loss_function.mse_loss(outputs, targets).item()
                epoch_losses["spectral"] += loss_function.spectral_loss(outputs, targets).item()
                epoch_losses["perceptual"] += loss_function.perceptual_loss(outputs, targets).item()
                epoch_losses["multiscale"] += loss_function.multi_scale_loss(outputs, targets).item()
                epoch_losses["combined"] += combined_loss.item()
                epoch_losses["count"] += 1

                Fine_tune_logger.info(
                    f"Epoch [{epoch+1}], Batch [{batch_idx}] -> Combined Loss: {combined_loss.item():.6f}, Mask Loss: {mask_loss.item():.6f}, Total Loss: {total_loss.item():.6f}"
                )

            avg_train_loss = running_loss / len(Fine_tuned_training_loader)
            Fine_tune_logger.info(f"Epoch [{epoch+1}/{fine_tune_epochs}] Average Training Loss: {avg_train_loss:.6f}")

            # Validation Phase
            model_engine.eval()
            running_val_loss = 0.0
            sdr_list, sir_list, sar_list = [], [], []
            with torch.no_grad():
                samples_visualized = 0  
                max_visualizations = 3  
                for val_batch_idx, (inputs, targets) in enumerate(Finetuned_validation_loader, start=1):
                    if inputs is None or targets is None:
                        Fine_tune_logger.warning(f"Skipping validation batch {val_batch_idx} due to None data.")
                        print(f"Skipping validation batch {val_batch_idx} due to None data.")
                        continue
                    inputs, targets = inputs.to(device), targets.to(device)
                    predicted_mask, outputs = model_engine(inputs)
                    predicted_mask = predicted_mask[:, :1, :, :]
                    outputs = outputs[:, :1, :, :]
                    combined_loss, _, _ = loss_function(predicted_mask, inputs, targets, outputs)
                    running_val_loss += combined_loss.item()
                    batch_sdr, batch_sir, batch_sar = evaluate_metrics_from_spectrograms(targets, outputs, loss_function)
                    sdr_list.extend(batch_sdr)
                    sir_list.extend(batch_sir)
                    sar_list.extend(batch_sar)

                    if samples_visualized < max_visualizations:
                        for sample_idx in range(inputs.size(0)):
                            target_np = targets[sample_idx].detach().cpu().numpy().flatten()
                            if np.allclose(target_np, 0):
                                Fine_tune_logger.info(f"Skipping visualization for silent target in sample {sample_idx+1} of batch {val_batch_idx}.")
                                continue
                            gt_waveform = loss_function.spectrogram_to_waveform(targets[sample_idx].unsqueeze(0), n_fft=1024, hop_length=512)[0]
                            pred_waveform = loss_function.spectrogram_to_waveform(outputs[sample_idx].unsqueeze(0), n_fft=1024, hop_length=512)[0]
                            visualize_and_save_waveforms(gt_waveform, pred_waveform, sample_idx + 1, epoch, visualization_dir)
                            samples_visualized += 1
                            if samples_visualized >= max_visualizations:
                                break

            avg_val_loss = running_val_loss / len(Finetuned_validation_loader)
            avg_sdr = sum(sdr_list) / len(sdr_list) if sdr_list else 0.0
            avg_sir = sum(sir_list) / len(sir_list) if sir_list else 0.0
            avg_sar = sum(sar_list) / len(sar_list) if sar_list else 0.0

            Fine_tune_logger.info(f"Validation Metrics - SDR: {avg_sdr:.4f}, SIR: {avg_sir:.4f}, SAR: {avg_sar:.4f}")
            Fine_tune_logger.info(f"Epoch [{epoch+1}/{fine_tune_epochs}] Average Validation Loss: {avg_val_loss:.6f}")

            Append_loss_values(loss_history_finetuning_epoches, epoch_losses, epoch)
            scheduler.step(avg_val_loss)
            Fine_tune_logger.debug(f"Scheduler stepped with validation loss: {avg_val_loss:.6f}")
            print(f"Scheduler stepped with validation loss: {avg_val_loss:.6f}")

        # Plot loss curves after training
        plot_loss_curves_FineTuning_script_(loss_history_finetuning_epoches, 'loss_curves_finetuning_epoches.png')
        Fine_tune_logger.info("Loss curves plotted.")
        # Save the fine-tuned model checkpoint via DeepSpeed
        checkpoint_dir = os.path.dirname(fine_tuned_model_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_engine.save_checkpoint(checkpoint_dir, tag="finetuned")
        Fine_tune_logger.info(f"Fine-tuned model checkpoint saved at {fine_tuned_model_path}")

    except Exception as e:
        Fine_tune_logger.error(f"Fine-tuning failed: {e}")
        Fine_tune_logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    fine_tune_model(
        fine_tuned_model_path=fine_tuned_model_path,
        Fine_tuned_training_loader=train_loader,
        Finetuned_validation_loader=val_loader_phase,
        ds_config=ds_config,
        fine_tune_epochs=8
    )
