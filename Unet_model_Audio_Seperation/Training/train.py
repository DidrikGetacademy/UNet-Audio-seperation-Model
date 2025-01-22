import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils as nn_utils
import os
import sys
import deepspeed
from torch import autocast, GradScaler
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Training.Externals.Dataloader import create_dataloaders
from Training.Externals.Value_storage import  Append_loss_values_for_batch, Append_loss_values_for_epoches, Get_calculated_average_loss_from_batches, loss_history_Epoches,loss_history_Batches
from Training.Externals.Logger import setup_logger,log_batch_losses,log_epoch_losses,logging_avg_loss_epoch,logging_avg_loss_batches,prev_epoch_loss_log,tensorboard_spectrogram_logging,log_first_2_batches_inputs_targets,log_first_2_batches_outputs_inputs_targets_predicted_mask
from Model_Architecture.model import UNet
from Training.Externals.Loss_Class_Functions import Combinedloss 
from Training.Externals.Memory_debugging import log_memory_after_index_epoch
from Training.Externals.Loss_Diagram_Values import  log_spectrograms_to_tensorboard, create_loss_diagrams
from Training.Externals.Functions import (
    Validate_epoch,
    training_completed,
    save_final_model,
    Model_Structure_Information,
    dataset_sample_information,
    check_inputs_targets_dataset,
    save_model_checkpoint,
    load_model_path_func,
    print_inputs_targets_shape,
    Early_break,
    Automatic_Fine_Tune,
    save_best_model,
    return_representive_batch,
)
log_dir = r"C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_Performance_logg\Tensorboard"  
train_logger = setup_logger( 'train', r'C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_performance_logg\log\Model_Training_logg.txt')
Final_model_path = r"C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_Weights\Pre_trained"
fine_tuned_model_base_path = r"C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_weights\Fine_tuned"
Model_CheckPoint = r"C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_Weights\CheckPoints"
MUSDB18_dir = r'C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Datasets\Dataset_Audio_Folders\musdb18'
DSD100_dataset_dir =r'C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Datasets\Dataset_Audio_Folders\DSD100'
os.makedirs(fine_tuned_model_base_path, exist_ok=True)
os.makedirs(Model_CheckPoint, exist_ok=True)
os.makedirs(Final_model_path, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)




#DEEPSPEED Configuration
ds_config = {
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 8,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": True,
        "overlap_comm": False,
        "reduce_scatter": False
    },
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "betas": [0.8, 0.9],
            "eps": 1e-6
        }
    },
    "scheduler": {
        "type": "ReduceLROnPlateau",
        "params": {
            "mode": "min",
            "factor": 0.5,
            "patience": 3,
            "threshold": 1e-3,
            "verbose": True,
            "cooldown": 1,
            "min_lr": 1e-5
        }
    },
    "steps_per_print": 50
}


#Training config
epochs = 2
patience = 2
best_loss = float('inf')
best_val_loss = float('inf')
trigger_times = 0
gradient_clip_value = 0.5 
prev_epoch_loss = None
best_model_path = None 
maskloss_avg, hybridloss_avg, combined_loss_avg = 0.0, 0.0, 0.0 
num_workers = 0
sampling_rate = 44100
max_length_seconds = 5
fine_tuning_flag = True












def train(load_model_path=r"C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_Weights\Fine_tuned\model.pth",start_training=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_logger.info(f"[Train] Using device: {device}")
  

    #UNet - Model initialization, Optimizer Config, Custom Hybridloss function, Gradscaler.
    model = UNet(in_channels=1, out_channels=1).to(device)

    #Initializing deepspeed
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters()
        config_params=ds_config

    )
    #Loss functionality
    criterion = Combinedloss()

    #Prininting Model Structure/Information
    Model_Structure_Information(model)

    # Loading model if exists or creating a new one. 
    load_model_path_func(load_model_path, model, device)
     
    if 'combined_train_loader' not in globals():
         combined_train_loader, combined_val_loader = create_dataloaders(
            musdb18_dir=MUSDB18_dir,
            dsd100_dir=DSD100_dataset_dir,
            batch_size=ds_config["train_micro_batch_size_per_gpu"],
            num_workers=num_workers,
            sampling_rate=sampling_rate,
            max_length_seconds=max_length_seconds,
            max_files_train=None,
            max_files_val=None,
    )

    #Prints sample of the dataset.
    dataset_sample_information(combined_train_loader,combined_val_loader)
 



#TRAINING LOOP STARTS HERE.....
    if start_training:
        train_logger.info(f"Training started with configuration: Batch size={ds_config['train_micro_batch_size_per_gpu']}, Effective batch size: {ds_config['train_batch_size']} Learning rate={ds_config['optimizer']['params']['lr']}, Epochs={epochs}")

        try:
            for epoch in range(epochs):
                model_engine.train()
                running_loss = 0.0
                representative_batch = None  
                train_logger.info(f"[Train] Epoch {epoch + 1}/{epochs} started.")

                for batch_idx, (inputs, targets) in enumerate(combined_train_loader, start=1):

                    log_first_2_batches_inputs_targets(batch_idx,train_logger,inputs,targets)
                    check_inputs_targets_dataset(inputs, targets, batch_idx)
                    print_inputs_targets_shape(inputs, targets, batch_idx)


                    #Moves the data to device.
                    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)




                   ###AUTOCAST###
                    with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                        predicted_mask, outputs = model_engine(inputs)
                        predicted_vocals = predicted_mask * inputs

                        representative_batch = return_representive_batch(inputs,targets,predicted_vocals,batch_idx)
                        log_first_2_batches_outputs_inputs_targets_predicted_mask(batch_idx,outputs,inputs,targets,predicted_mask,train_logger)


                        combined_loss, mask_loss, hybrid_loss = criterion(predicted_mask, inputs, targets)


                        
                       


                    #LOGGING & APPENDING values
                    global_step = epoch * len(combined_train_loader) + batch_idx
               
                    log_batch_losses(global_step, combined_loss, mask_loss, hybrid_loss)

                    Append_loss_values_for_batch(mask_loss, hybrid_loss, combined_loss)

                    
                    model_engine.backward(combined_loss)
                    model_engine.step()


                    train_logger.info(
                        f"[Batch] Epoch: {epoch + 1}/{epochs}, Batch: {batch_idx}/{len(combined_train_loader)}, "
                        f"Combined Loss: {combined_loss.item():.6f}, Mask Loss: {mask_loss.item():.6f}, Hybrid Loss: {hybrid_loss.item():.6f}"
                    )

                    # Check for NaN/Inf in loss
                    if torch.isnan(combined_loss) or torch.isinf(combined_loss):
                        train_logger.error(f"Skipping Batch {batch_idx} due to NaN/Inf in loss.")
                        continue

                    # Check for invalid outputs
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        train_logger.error(f"Skipping Batch {batch_idx} due to NaN/Inf in outputs.")
                        continue

                    running_loss += combined_loss.item()

      
                tensorboard_spectrogram_logging(representative_batch, log_spectrograms_to_tensorboard, epoch)

            
                scheduler.step(running_loss)

                # Current Learning Rate logging
                current_lr = optimizer.param_groups[0]['lr']
                train_logger.info(f"Current Learning Rate {current_lr}")

                # Average epoch loss
                avg_epoch_loss = running_loss / len(combined_train_loader)
                train_logger.info(f"[Epoch Summary] Avg Training Loss: {avg_epoch_loss:.6f}")

                # Logging average loss
                prev_epoch_loss_log(train_logger, prev_epoch_loss, avg_epoch_loss, epoch)

                # Save model checkpoint if loss improved
                best_loss, trigger_times = save_model_checkpoint(avg_epoch_loss, epoch, model_engine, best_loss, trigger_times)

                # Validation
                if len(combined_val_loader) > 0:
                    avg_combined_loss, avg_mask_loss, avg_hybrid_loss = Validate_epoch(model_engine, combined_val_loader, criterion, device)
                    if avg_combined_loss < best_val_loss:
                        best_val_loss = avg_combined_loss
                        val_checkpoint_path = os.path.join(Model_CheckPoint, f"best_val_model_epoch_{epoch + 1}.pth")
                        torch.save(model_engine.state_dict(), val_checkpoint_path)
                        train_logger.info(f"Saved best validation model at epoch {epoch + 1} with loss {best_val_loss:.6f}")

                    logging_avg_loss_batches(train_logger, epoch, epochs, avg_combined_loss, avg_mask_loss, avg_hybrid_loss)

                # Calculate average losses for epoch
                maskloss_avg, hybridloss_avg, combined_loss_avg = Get_calculated_average_loss_from_batches()

                # Tensorboard logging for epoch losses
                log_epoch_losses(epoch, combined_loss_avg, maskloss_avg, hybridloss_avg)
                Append_loss_values_for_epoches(maskloss_avg, hybridloss_avg, combined_loss_avg, avg_epoch_loss)

                # Early stopping
                if Early_break(trigger_times, patience, train_logger):
                    break

                # Logging memory after each epoch
                log_memory_after_index_epoch(epoch)

                # Logging average loss per epoch
                logging_avg_loss_epoch(epoch, avg_epoch_loss, maskloss_avg, hybridloss_avg, train_logger)
                 
                prev_epoch_loss = avg_epoch_loss
                Append_loss_values_for_epoches(maskloss_avg, prev_epoch_loss, epochs, hybridloss_avg, combined_loss_avg, avg_epoch_loss)

            # Create loss diagrams after training
            create_loss_diagrams(loss_history_Batches, loss_history_Epoches)      

            # Training completion
            training_completed()

            # Save the best and final models
            save_best_model(model_engine, best_model_path, Final_model_path, train_logger)
            save_final_model(model_engine, Final_model_path)

            # Fine-tuning if flag is set
            if fine_tuning_flag:
                Automatic_Fine_Tune(
                    combined_val_loader,
                    combined_train_loader,
                    fine_tuned_model_base_path,
                    Final_model_path
                )
        
        except Exception as e:
            train_logger.error(f"[Train] Error during training: {str(e)}")
    else:
        train_logger.info("Training was not started.")

if __name__ == "__main__":

    train(load_model_path=r"C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_Weights\Fine_tuned\model.pth")
    
    

