# fine_tuned_model.py
from mir_eval.separation import bss_eval_sources
import os
import sys
import torch
import librosa
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autocast, GradScaler
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Training.Externals.Logger import setup_logger
from Training.Externals.Dataloader import create_dataloaders
from Model_Architecture.model import UNet
from Training.Loss_Diagram_Values import plot_loss_curves_FineTuning_script_  

# Initialize Logger
Fine_tune_logger = setup_logger(
    'Fine-Tuning',
    r'C:\Users\didri\Desktop\UNet Models\UNet_vocal_isolation_model\Model_performance_logg\log\Model_Training_logg.txt'
)

# Global loss history
loss_history_finetuning_epoches = {
    "l1": [],
    "mse": [],
    "spectral": [],
    "perceptual": [],
    "multiscale": [],
    "combined": [],
}

def Append_loss_values(loss_history, total_loss, l1_val, multi_scale_val, perceptual_val, spectral_val, mse_val, epoch):
    loss_history["l1"].append(l1_val.item())
    loss_history["combined"].append(total_loss.item())
    loss_history["spectral"].append(spectral_val.item())
    loss_history["perceptual"].append(perceptual_val.item())
    loss_history["multiscale"].append(multi_scale_val.item())
    loss_history["mse"].append(mse_val.item())

    # Logging
    Fine_tune_logger.info(
        f"[Fine-Tune] Epoch {epoch+1} "
        f"L1 Loss={l1_val.item():.6f}, MSE Loss={mse_val.item():.6f}, "
        f"Spectral Loss={spectral_val.item():.6f}, Perceptual Loss={perceptual_val.item():.6f}, "
        f"Multi-Scale Loss={multi_scale_val.item():.6f}, Combined Loss={total_loss.item():.6f}"
    )

def spectrogram_to_waveform(spectrogram, n_fft=2048, hop_length=512):
    """
    Converts a spectrogram (tensor) back to an audio waveform using Griffin-Lim.
    Args:
        spectrogram: Tensor of shape [batch, channels, freq_bins, time_steps]
        n_fft: FFT window size
        hop_length: Hop size between FFT windows
    Returns:
        waveforms: List of reconstructed waveforms
    """
    spectrogram_np = spectrogram.detach().cpu().numpy()  # Convert tensor to numpy
    assert len(spectrogram_np.shape) == 4, (
        f"Expected spectrogram shape [batch, channels, freq_bins, time_steps], "
        f"but got {spectrogram_np.shape}."
    )
    assert spectrogram_np.shape[1] == 1, (
        f"Expected single-channel spectrograms with shape [batch, 1, freq_bins, time_steps], "
        f"but got {spectrogram_np.shape[1]} channels."
    )

    waveforms = []
    for i in range(spectrogram_np.shape[0]):  # Loop over batch
        mag = spectrogram_np[i, 0, :, :]  # Explicit indexing
        assert len(mag.shape) == 2, (
            f"Expected magnitude shape [freq_bins, time_steps], but got {mag.shape}."
        )
        waveform = librosa.griffinlim(mag, n_iter=32, hop_length=hop_length, win_length=n_fft)
        waveforms.append(waveform)

    return waveforms
def evaluate_metrics_from_spectrograms(ground_truth, predicted, n_fft=2048, hop_length=512):
    """
    Evaluates SDR, SIR, and SAR metrics for spectrogram inputs.
    Skips evaluation if reference source (ground_truth) is silent (all zeros).
    """
    gt_waveforms = spectrogram_to_waveform(ground_truth, n_fft, hop_length)
    pred_waveforms = spectrogram_to_waveform(predicted, n_fft, hop_length)

    sdr_list, sir_list, sar_list = [], [], []
    for gt, pred in zip(gt_waveforms, pred_waveforms):
        if np.allclose(gt, 0):  # Skip silent references
            print("Skipping evaluation for a silent reference.")
            continue
        gt = gt[:len(pred)]  # Ensure the two waveforms have the same length
        pred = pred[:len(gt)]
        sdr, sir, sar, _ = bss_eval_sources(gt[np.newaxis, :], pred[np.newaxis, :])  # Ensure correct shape
        sdr_list.append(sdr[0])
        sir_list.append(sir[0])
        sar_list.append(sar[0])

    return sdr_list, sir_list, sar_list





def freeze_encoder(model):

    for layer in model.encoder:
        for param in layer.parameters():
            param.requires_grad = False
    model.encoder.eval()
    Fine_tune_logger.info("Encoder layers frozen for fine-tuning.")


class HybridLoss:
    def __init__(self, device):
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.device = device

        Fine_tune_logger.info("Loading VGGish model...")
        self.audio_model = torch.hub.load('harritaylor/torchvggish', 'vggish', trust_repo=True).to(device)
        self.audio_model.eval()
        Fine_tune_logger.info("Fine-tuning --> VGGish model loaded and set to eval mode.")

        for param in self.audio_model.parameters():
            param.requires_grad = False

    def spectral_loss(self, output, target):
        loss_value = torch.mean((output - target) ** 2)
        Fine_tune_logger.info(f"Fine-tuning --> [Spectral Loss] {loss_value.item():.6f}")
        return loss_value

    def spectrogram_batch_to_audio_list(self, spectrogram_batch, n_fft=2048, hop_length=512):
        spect_np = spectrogram_batch.detach().cpu().numpy()
        waveforms = []

        Fine_tune_logger.debug(f"Fine-tuning --> [Spectrogram to Audio] Spectrogram shape: {spectrogram_batch.shape}")

        for b in range(spect_np.shape[0]):
            single_mag = spect_np[b, 0]
            audio = librosa.griffinlim(
                single_mag,
                n_iter=32,
                hop_length=hop_length,
                win_length=n_fft
            )
            waveforms.append(torch.tensor(audio, device=self.device, dtype=torch.float32))

        return waveforms

    def perceptual_loss(self, output, target):
        output_waveforms = self.spectrogram_batch_to_audio_list(output)
        target_waveforms = self.spectrogram_batch_to_audio_list(target)

        batch_size = len(output_waveforms)
        orig_sr, new_sr = 44100, 16000
        max_length = 176400

        output_audio_list, target_audio_list = [], []

        for i in range(batch_size):
            out_np = output_waveforms[i].cpu().numpy()
            tgt_np = target_waveforms[i].cpu().numpy()

            out_16k = librosa.resample(out_np, orig_sr=orig_sr, target_sr=new_sr)
            tgt_16k = librosa.resample(tgt_np, orig_sr=orig_sr, target_sr=new_sr)

            if len(out_16k) < max_length:
                out_16k = np.pad(out_16k, (0, max_length - len(out_16k)))
            else:
                out_16k = out_16k[:max_length]

            if len(tgt_16k) < max_length:
                tgt_16k = np.pad(tgt_16k, (0, max_length - len(tgt_16k)))
            else:
                tgt_16k = tgt_16k[:max_length]

            output_audio_list.append(out_16k)
            target_audio_list.append(tgt_16k)

        output_audio_np = np.stack(output_audio_list, axis=0)
        target_audio_np = np.stack(target_audio_list, axis=0)

        # Extract features from VGGish
        output_features_list, target_features_list = [], []
        with torch.no_grad():
            for i in range(batch_size):
                out_feat = self.audio_model(output_audio_np[i], fs=new_sr)
                tgt_feat = self.audio_model(target_audio_np[i], fs=new_sr)
                output_features_list.append(out_feat)
                target_features_list.append(tgt_feat)

        output_features = torch.cat(output_features_list, dim=0)
        target_features = torch.cat(target_features_list, dim=0)
        loss_value = self.mse_loss(output_features, target_features)
        Fine_tune_logger.debug(f"Fine-tuning --> [Perceptual Loss] {loss_value.item():.6f}")

        return loss_value

    def multi_scale_loss(self, output, target, scales=[1, 2, 4]):
        total_multi_scale_loss = 0.0
        for scale in scales:
            scaled_output = F.avg_pool2d(output, kernel_size=scale)
            scaled_target = F.avg_pool2d(target, kernel_size=scale)
            scale_loss = self.mse_loss(scaled_output, scaled_target)
            Fine_tune_logger.debug(f"[Multi-Scale Loss] Scale={scale}, Loss={scale_loss.item():.6f}")
            total_multi_scale_loss += scale_loss
        Fine_tune_logger.debug(f"[Multi-Scale Loss] Total Loss for scales {scales}: {total_multi_scale_loss.item():.6f}")

        return total_multi_scale_loss

    def combined_loss(self, output, target):
        l1 = self.l1_loss(output, target)
        mse = self.mse_loss(output, target)
        spectral = self.spectral_loss(output, target)
        perceptual = self.perceptual_loss(output, target)
        multi_scale = self.multi_scale_loss(output, target)

        total_loss = (
            0.3 * l1 +
            0.3 * mse +
            0.1 * spectral +
            0.2 * perceptual +
            0.1 * multi_scale
        )
        Fine_tune_logger.debug(
            f"Fine-tuning --> [Combined Loss] L1={l1.item():.6f}, MSE={mse.item():.6f}, "
            f"Spectral={spectral.item():.6f}, Perceptual={perceptual.item():.6f}, "
            f"Multi-Scale={multi_scale.item():.6f}, Total={total_loss.item():.6f}"
        )

        return total_loss, l1, mse, spectral, perceptual, multi_scale




def fine_tune_model( pretrained_model_path, fine_tuned_model_path, combined_train_loader, combined_val_loader,  learning_rate=1e-7,   fine_tune_epochs=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Fine_tune_logger.info(f"Fine-tuning --> Using device: {device}")

    Fine_tune_logger.info("Initializing model...")
    model = UNet(in_channels=1, out_channels=1).to(device)

    if pretrained_model_path is None:
        Fine_tune_logger.error("Pretrained model path must be provided for fine-tuning.")
        raise ValueError("Pretrained model path is None.")

    # Load pretrained model weights
    state_dict = torch.load(pretrained_model_path, map_location=device,weights_only=True)
    model.load_state_dict(state_dict)
    Fine_tune_logger.info(f"Fine-tuning --> Pretrained model loaded from: {pretrained_model_path}")

    # Freeze encoder layers
    freeze_encoder(model)
    Fine_tune_logger.info("Fine-tuning --> Encoder layers frozen.")

    # Initialize loss functions and optimizer
    loss_functions = HybridLoss(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=2,
        factor=0.5
    )

    scaler = GradScaler()

    Fine_tune_logger.info(f"Fine-tuning --> Starting fine-tuning: Epochs={fine_tune_epochs}, LR={learning_rate}")
    print("Starting fine-tuning...")

    try:
        for epoch in range(fine_tune_epochs):
            # Training Phase
            model.train()
            running_loss = 0.0

            for batch_idx, (inputs,targets) in enumerate(combined_train_loader, start=1):


                
                if inputs is None or targets is None:
                    Fine_tune_logger.warning(f"Skipping training batch {batch_idx} due to None data.")
                    print(f"Skipping training batch {batch_idx} due to None data.")
                    continue

                inputs, targets = inputs.to(device), targets.to(device)

                Fine_tune_logger.debug(
                    f"[Fine-Tune] Epoch {epoch+1}, Training Batch {batch_idx} | "
                    f"Inputs: {inputs.shape}, {inputs.min():.4f}-{inputs.max():.4f} | "
                    f"Targets: {targets.shape}, {targets.min():.4f}-{targets.max():.4f}"
                )
 

                optimizer.zero_grad() 

                with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                    outputs = model(inputs)
                    Fine_tune_logger.debug(
                        f"[Fine-Tune] Epoch {epoch+1}, Training Batch {batch_idx} | "
                        f"Outputs: {outputs.shape}, {outputs.min():.4f}-{outputs.max():.4f}"
                    )

                    total_loss, l1_val, mse_val, spectral_val, perceptual_val, multi_scale_val = loss_functions.combined_loss(outputs, targets)

                Append_loss_values(
                    loss_history_finetuning_epoches,
                    total_loss,
                    l1_val,
                    multi_scale_val,
                    perceptual_val,
                    spectral_val,
                    mse_val,
                    epoch
                )

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += total_loss.item()
                Fine_tune_logger.info(
                    f"Fine-Tuning --> [Epoch {epoch+1}, Training Batch {batch_idx}] "
                    f"L1={l1_val.item():.6f}, MSE={mse_val.item():.6f}, "
                    f"Spectral={spectral_val.item():.6f}, Perceptual={perceptual_val.item():.6f}, "
                    f"Multi-Scale={multi_scale_val.item():.6f}, TotalLoss={total_loss.item():.6f}"
                )

            # Calculate average training loss
            avg_train_loss = running_loss / len(combined_train_loader)
            Fine_tune_logger.info(f"Fine-Tuning --> Epoch [{epoch+1}/{fine_tune_epochs}] Average Training Loss: {avg_train_loss:.6f}")
            print(f"Fine-Tuning --> Epoch [{epoch+1}/{fine_tune_epochs}] Average Training Loss: {avg_train_loss:.6f}")

            # Validation Phase
            model.eval()
            running_val_loss = 0.0
            
            with torch.no_grad():
                for batch_idx, data in enumerate(combined_val_loader, start=1):
                    if data is None:
                        Fine_tune_logger.warning(f"Skipping validation batch {batch_idx} due to None data.")
                        print(f"Skipping validation batch {batch_idx} due to None data.")
                        continue

                    inputs, targets = data
                    if inputs is None or targets is None:
                        Fine_tune_logger.warning(f"Skipping validation batch {batch_idx} due to None data.")
                        print(f"Skipping validation batch {batch_idx} due to None data.")
                        continue

                    inputs, targets = inputs.to(device), targets.to(device)

                    Fine_tune_logger.debug(
                        f"[Fine-Tune] Epoch {epoch+1}, Validation Batch {batch_idx} | "
                        f"Inputs: {inputs.shape}, {inputs.min():.4f}-{inputs.max():.4f} | "
                        f"Targets: {targets.shape}, {targets.min():.4f}-{targets.max():.4f}"
                    )
                    print(
                        f"[Fine-Tune] Epoch {epoch+1}, Validation Batch {batch_idx} | "
                        f"Inputs: {inputs.shape}, {inputs.min():.4f}-{inputs.max():.4f} | "
                        f"Targets: {targets.shape}, {targets.min():.4f}-{targets.max():.4f}"
                    )

                    with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                        outputs = model(inputs)
                        total_loss, _, _, _, _, _ = loss_functions.combined_loss(outputs, targets)

                    running_val_loss += total_loss.item()
                    Fine_tune_logger.info(
                        f"Fine-Tuning --> [Epoch {epoch+1}, Validation Batch {batch_idx}] "
                        f"TotalLoss={total_loss.item():.6f}"
                    )

            # Calculate average validation loss
            avg_val_loss = running_val_loss / len(combined_val_loader)
            Fine_tune_logger.info(f"Fine-Tuning --> Epoch [{epoch+1}/{fine_tune_epochs}] Average Validation Loss: {avg_val_loss:.6f}")
            print(f"Fine-Tuning --> Epoch [{epoch+1}/{fine_tune_epochs}] Average Validation Loss: {avg_val_loss:.6f}")

            # Scheduler Step with validation loss
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            sdr_list, sir_list, sar_list = [], [], []
            for batch_idx, (inputs, targets) in enumerate(combined_val_loader, start=1):
                if inputs is None or targets is None:
                    continue

                inputs, targets = inputs.to(device), targets.to(device)
                with torch.no_grad():
                    outputs = model(inputs)

                batch_sdr, batch_sir, batch_sar = evaluate_metrics_from_spectrograms(targets, outputs)
                sdr_list.extend(batch_sdr)
                sir_list.extend(batch_sir)
                sar_list.extend(batch_sar)

            # Log average metrics
            avg_sdr = sum(sdr_list) / len(sdr_list)
            avg_sir = sum(sir_list) / len(sir_list)
            avg_sar = sum(sar_list) / len(sar_list)

            Fine_tune_logger.info(f"Validation Metrics - Avg SDR: {avg_sdr:.4f}->[Måler hvor godt det isolerte signalet samsvarer med det opprinnelige målsignalet, med tanke på alle typer feil (støy, interferens, og kunstige artefakter Høyere SDR er bedre[God-ytelse: SDR > 10 db,   akseptabel SDR 6db eller større. dårlig ytelse: sdr < 6])]")
            Fine_tune_logger.info(f" Avg SIR: {avg_sir:.4f}--> [Måler hvor godt modellen separerer målsignalet fra andre kilder (interferens).Høyere SIR er bedre. .... God ytelse: SIR > 15 dB.... Akseptabel ytelse: 10 dB ≤ SIR ≤ 15 db..... Dårlig ytelse: SIR < 10 dB]")
            Fine_tune_logger.info(f" Avg SAR: {avg_sar:.4f}--> [Måler hvor mye artefakter modellen introduserer under isoleringsprosessen.]....God ytelse: SAR > 10 dB, Akseptabel ytelse: 7 dB ≤ SAR ≤ 10 dB, Dårlig ytelse: SAR < 7 dB")

            Fine_tune_logger.info(f"Fine-Tuning --> Epoch [{epoch+1}] Completed. Current LR: {current_lr:e}")
            print(f"Fine-Tuning --> Epoch [{epoch+1}] Completed. Current LR: {current_lr:e}")

            # Plot loss curves
        plot_loss_curves_FineTuning_script_(loss_history_finetuning_epoches, 'loss_curves_finetuning_epoches.png')

    except Exception as e:
        Fine_tune_logger.error(f"Fine-Tuning --> Error during fine-tuning: {str(e)}")
        raise e

    # Save the fine-tuned model
    torch.save(model.state_dict(), fine_tuned_model_path)
    Fine_tune_logger.info(f"Fine-Tuning --> Fine-tuned model saved to: {fine_tuned_model_path}")
    print(f"Fine-tuning completed. Model saved to {fine_tuned_model_path}.")
