import os
import sys
import torch
import deepspeed
import numpy as np
import traceback
from torch import autocast
import json 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Training.Externals.utils import Return_root_dir
from Training.Externals.Loss_Class_Functions import Combinedloss_Fine_tuning 
from Training.Externals.Logger import setup_logger
from Training.Externals.Dataloader import create_dataloader_Fine_tuning
from Training.Externals.Loss_Diagram_Values import plot_loss_curves_FineTuning_script_,visualize_and_save_waveforms,evaluate_metrics_from_spectrograms
from Training.Externals.Functions import load_model_path_func
from Training.Externals.Functions import freeze_encoder
from Training.Externals.Value_storage import Append_loss_values_epoches,get_loss_value_list_loss_history_finetuning_epoches
root_dir = Return_root_dir() 
fine_tuned_model_path = os.path.join(root_dir, "Model_Weights/Fine_Tuned/Model.pth")
Fine_tune_path = os.path.join(root_dir, "DeepSeed_Configuration/ds_config_Training.json")
Fine_tune_logger = setup_logger('Fine-Tuning', Fine_tune_path)

with open(os.path.join(root_dir, "DeepSeed_Configuration/ds_config_Training.json"), "r") as f:
    ds_config = json.load(f)

torch.backends.cudnn.benchmark = True 
torch.set_num_threads(8)

loss_history_finetuning_epoches = {
    "l1": [],
    "mse": [],
    "spectral": [],
    "perceptual": [],
    "multiscale": [],
    "combined": [],
}


if 'fine_tuned_loader' not in globals():
        eval_loader,Fine_tuned_training_loader = create_dataloader_Fine_tuning(
            batch_size=ds_config["train_micro_batch_size_per_gpu"], 
    )





def fine_tune_model(fine_tuned_model_path, Fine_tuned_training_loader,Finetuned_validation_loader, model_engine, fine_tune_epochs=10, pretrained_model_path=None):
    visualization_dir = os.path.join(os.path.dirname(fine_tuned_model_path), "visualizations")
    from Model_Architecture.model import UNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Fine_tune_logger.info(f"Fine-tuning --> Using device: {device}")
    load_model_path = os.path.join(root_dir, "Model_Weights/CheckPoints/checkpoint_epochsss_25")
    Fine_tune_logger.info("Initializing model...")
    
    model = UNet(in_channels=1, out_channels=1).to(device)


    
    if pretrained_model_path is None:
        load_model_path_func(load_model_path, model_engine, model, device)
    else:
        load_model_path_func(pretrained_model_path, model_engine, model, device)
    

    Fine_tune_logger.debug("DeepSpeed model_engine initialized.")
    
    
    freeze_encoder(Fine_tune_logger, model_engine)

    loss_function = Combinedloss_Fine_tuning(Fine_tune_logger,device)
    
    

    try:
        for epoch in range(fine_tune_epochs):
            Fine_tune_logger.info(f"Starting Epoch {epoch+1}/{fine_tune_epochs}")
            print(f"Starting Epoch {epoch+1}/{fine_tune_epochs}")
            model_engine.train()
            running_loss = 0.0

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

                    predicted_mask = predicted_mask[:, :1, :, :]
                    outputs = outputs[:, :1, :, :]
                    combined_loss, mask_loss, total_loss = loss_function(predicted_mask, inputs, targets, outputs)
                model_engine.backward(combined_loss)
                model_engine.step()

                running_loss += combined_loss.item()

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
                    batch_sdr, batch_sir, batch_sar = evaluate_metrics_from_spectrograms(Fine_tune_logger, targets, outputs, loss_function)
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
                            visualize_and_save_waveforms(Fine_tune_logger,gt_waveform, pred_waveform, sample_idx + 1, epoch, visualization_dir)
                            samples_visualized += 1
                            if samples_visualized >= max_visualizations:
                                break



            avg_val_loss = running_val_loss / len(Finetuned_validation_loader)
            avg_sdr = sum(sdr_list) / len(sdr_list) if sdr_list else 0.0
            avg_sir = sum(sir_list) / len(sir_list) if sir_list else 0.0
            avg_sar = sum(sar_list) / len(sar_list) if sar_list else 0.0
            Fine_tune_logger.info(f"Validation Metrics - SDR: {avg_sdr:.4f}, SIR: {avg_sir:.4f}, SAR: {avg_sar:.4f}")
            Fine_tune_logger.info(f"Epoch [{epoch+1}/{fine_tune_epochs}] Average Validation Loss: {avg_val_loss:.6f}")

            loss_history_finetuning_epoches = get_loss_value_list_loss_history_finetuning_epoches

            scheduler.step(avg_val_loss)
            Fine_tune_logger.info(f"Scheduler stepped with validation loss: {avg_val_loss:.6f}")



        Append_loss_values_epoches(Fine_tune_logger,loss_history_finetuning_epoches, epoch_losses, epoch)


        # Plot loss curves after training
        plot_loss_curves_FineTuning_script_(loss_history_finetuning_epoches, 'loss_curves_finetuning_epoches.png')
        Fine_tune_logger.info("Loss curves plotted.")
        
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
        Fine_tuned_training_loader = Fine_tuned_training_loader,
        Finetuned_validation_loader = eval_loader,
        ds_config=ds_config
    )