import torch
import os
import sys
import deepspeed
from torch import autocast
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
print("Current working directory:", os.getcwd())
print("Current sys.path:", sys.path)
print(f"project root {project_root}")
from Training.Externals.Dataloader import create_dataloaders
from Training.Externals.Value_storage import  Append_loss_values_for_batch, Append_loss_values_for_epoches, Get_calculated_average_loss_from_batches, loss_history_Epoches,loss_history_Batches
from Training.Externals.Logger import setup_logger,log_batch_losses,log_epoch_losses,logging_avg_loss_epoch,logging_avg_loss_batches,prev_epoch_loss_log,tensorboard_spectrogram_logging,log_first_2_batches_inputs_targets,log_first_2_batches_outputs_inputs_targets_predicted_mask
from Model_Architecture.model import UNet
from Training.Externals.Loss_Class_Functions import Combinedloss 
from Training.Externals.Memory_debugging import log_memory_after_index_epoch
from Training.Externals.Loss_Diagram_Values import  log_spectrograms_to_tensorboard, create_loss_diagrams
from Training.Evaluation_Basic import Validate_epoch

from Training.Externals.Functions import (
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
    Return_root_dir,
)

#PATHS
root_dir = Return_root_dir() #Gets the root directory
TensorBoard_log_dir = os.path.join(root_dir, "Model_Performance_logg/Tensorboard")
fine_tuned_model_base_path = os.path.join(root_dir, "Model_weights/Fine_tuned")
Model_CheckPoint = os.path.join(root_dir, "Model_Weights/CheckPoints")
Final_model_path = os.path.join(root_dir, "Model_Weights/Pre_trained")
MUSDB18_dir = os.path.join(root_dir, "Datasets/Dataset_Audio_Folders/musdb18")
DSD100_dataset_dir = os.path.join(root_dir, "Datasets/Dataset_Audio_Folders/DSD100")
train_log_path = os.path.join(root_dir, "Model_performance_logg/log/Model_Training_logg.txt")

#Checks if dirs exists
os.makedirs(fine_tuned_model_base_path, exist_ok=True)
os.makedirs(Model_CheckPoint, exist_ok=True)
os.makedirs(Final_model_path, exist_ok=True)
os.makedirs(TensorBoard_log_dir, exist_ok=True)


#Training Logger
train_logger = setup_logger('train', train_log_path)



ds_config_path = os.path.join(root_dir, "Training/ds.config.json")

# Load DeepSpeed config (You can load and modify it dynamically here)
import json
with open(ds_config_path, 'r') as f:
    ds_config = json.load(f)

# Get world size dynamically (number of available GPUs)
world_size = torch.cuda.device_count()  # or set it manually if needed
train_batch_size = ds_config["train_batch_size"]

# Dynamically calculate micro_batch_size and gradient_accumulation_steps
train_micro_batch_size_per_gpu = train_batch_size // world_size
gradient_accumulation_steps = train_batch_size // (train_micro_batch_size_per_gpu * world_size)

# Ensure this dynamic calculation makes sense and aligns
assert train_micro_batch_size_per_gpu * gradient_accumulation_steps * world_size == train_batch_size, \
    f"Inconsistent batch sizes: {train_micro_batch_size_per_gpu} * {gradient_accumulation_steps} * {world_size} != {train_batch_size}"

# Update DeepSpeed config dynamically
ds_config["train_micro_batch_size_per_gpu"] = train_micro_batch_size_per_gpu
ds_config["gradient_accumulation_steps"] = gradient_accumulation_steps
ds_config["world_size"] = world_size  # or set manually if needed




#Training config
load_model_path=None
epochs = 10
patience = 3
best_loss = float('inf')
best_val_loss = float('inf')
trigger_times = 0
gradient_clip_value = 0.5 
prev_epoch_loss = None
best_model_path = None 
maskloss_avg, hybridloss_avg, combined_loss_avg = 0.0, 0.0, 0.0 
num_workers = 0
sampling_rate = 44100
max_length_seconds = 15
fine_tuning_flag = False










def train(load_model_path,start_training=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_logger.info(f"[Train] Using device: {device}")
  

    #UNet - Model initialization, Optimizer Config, Custom Hybridloss function, Gradscaler.
    model = UNet(in_channels=1, out_channels=1)
    print(f"Loading model to {device}")
    model = model.to(device)
    print(f"Model loaded to {device}")

    #Initializing deepspeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config_params=ds_config
        )

        
    base_optimizer = optimizer.optimizer    # Get the base PyTorch optimizer wrapped by DeepSpeed
    scheduler = torch.optim.lr_scheduler.StepLR(base_optimizer, step_size=10, gamma=0.1)

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
            batch_size=ds_config["train_micro_batch_size_per_gpu"],  # Pass the batch size from DeepSpeed config
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
                    print(f"[Train] Moving batch {batch_idx} data to {device}...")

                    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                    print(f"[Train] Batch {batch_idx} moved to {device}.")




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
                        v.error(f"Skipping Batch {batch_idx} due to NaN/Inf in outputs.")
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

    train(load_model_path)
    
    

