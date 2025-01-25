import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
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
from Training.Externals.Logger import setup_logger,logging_avg_loss_epoch,logging_avg_loss_batches,prev_epoch_loss_log,log_first_2_batches_inputs_targets,log_first_2_batches_outputs_inputs_targets_predicted_mask
from Model_Architecture.model import UNet
from Training.Externals.Loss_Class_Functions import Combinedloss 
from Training.Externals.Memory_debugging import log_memory_after_index_epoch,clear_memory_before_training
from Training.Externals.Loss_Diagram_Values import  create_loss_diagrams
import logging
from Training.Externals.Functions import (
    training_completed,
    Model_Structure_Information,
    dataset_sample_information,
    check_inputs_targets_dataset,
    load_model_path_func,
    print_inputs_targets_shape,
    Early_break,
    Automatic_Fine_Tune,
    save_best_model,
    Return_root_dir,
    save_representative_batch,
)

#PATHS
root_dir = Return_root_dir() #Gets the root directory
fine_tuned_model_base_path = os.path.join(root_dir, "Model_weights/Fine_tuned")
Model_CheckPoint = os.path.join(root_dir, "Model_Weights/CheckPoints")
Final_model_path = os.path.join(root_dir, "Model_Weights/Pre_trained")
MUSDB18_dir = os.path.join(root_dir, "Datasets/Dataset_Audio_Folders/musdb18")
DSD100_dataset_dir = os.path.join(root_dir, "Datasets/Dataset_Audio_Folders/DSD100")
custom_dataset_dir = os.path.join(root_dir,"Datasets/Dataset_Audio_Folders/Custom_Dataset")
train_log_path = os.path.join(root_dir, "Model_performance_logg/log/Model_Training_logg.txt")

#Checks if dirs exists
os.makedirs(fine_tuned_model_base_path, exist_ok=True)
os.makedirs(Model_CheckPoint, exist_ok=True)
os.makedirs(Final_model_path, exist_ok=True)



#Training Logger
train_logger = setup_logger('train', train_log_path,level=logging.INFO)
print(f"Log file path: {train_log_path}")


import json
# Load the JSON configuration
with open(os.path.join(root_dir, "Training/ds.config.json"), "r") as f:
    ds_config = json.load(f)






def train(start_training=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_logger.info(f"[Train] Using device: {device}")
    load_model_path = os.path.join(Model_CheckPoint, "checkpoint_epochsss_1")
    patience = 5
    epochs = 40
    best_loss = float('inf')
    trigger_times = 0
    prev_epoch_loss = None
    best_model_path = None 
    maskloss_avg, hybridloss_avg, combined_loss_avg = 0.0, 0.0, 0.0 
    num_workers = 6
    sampling_rate = 44100
    max_length_seconds = 10
    fine_tuning_flag = True



    clear_memory_before_training()
    #UNet - Model initialization, Optimizer Config, Custom Hybridloss function, Gradscaler.
    model = UNet(in_channels=1, out_channels=1).to(device)



    #Initializing deepspeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config_params=ds_config
        )
    load_model_path_func(load_model_path, model_engine, model, device)

        


    #Loss functionality
    criterion = Combinedloss()

    #Prininting Model Structure/Information
    Model_Structure_Information(model)

    # Loading model if exists or creating a new one. 
     
    representative_batch = None  
    if 'combined_train_loader' not in globals():
         combined_train_loader, combined_val_loader = create_dataloaders(
            custom_dataset_dir,
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

        try:
            for epoch in range(epochs):
                model_engine.train()
                running_loss = 0.0
                train_logger.info(f"[Train] Epoch {epoch + 1}/{epochs} started.")
                print(f"[Train] Epoch {epoch + 1}/{epochs} started.")



                for batch_idx, (inputs, targets) in enumerate(combined_train_loader, start=1):

                    #Moves the data to device. 
                    print(f"[Train] Moving batch {batch_idx} data to {device}...")
                    inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
                    targets = targets.to(device, dtype=torch.float32, non_blocking=True)


                    if batch_idx <= 1:
                        print(f"Sample {batch_idx}: input min={inputs.min()}, max={inputs.max()}, target min={targets.min()}, max={targets.max()}")
                        train_logger.info(f"Sample {batch_idx}: input min={inputs.min()}, max={inputs.max()}, target min={targets.min()}, max={targets.max()}")
                    print(f"[Batch {batch_idx}] Batch size: {inputs.size(0)}")


                    
                    if inputs.max() == 0 or targets.max() == 0:
                        print(f"Skipping batch {batch_idx} due to zero inputs or targets.")
                        continue


                    log_first_2_batches_inputs_targets(batch_idx,train_logger,inputs,targets)
                    check_inputs_targets_dataset(inputs, targets, batch_idx)
                    print_inputs_targets_shape(inputs, targets, batch_idx)



              




                   ###AUTOCAST###
                    optimizer.zero_grad()
                    with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                        predicted_mask, outputs = model_engine(inputs)
                        predicted_vocals = predicted_mask * inputs

                        if batch_idx <= 2:
                             representative_batch = (inputs.detach().cpu(), predicted_vocals.detach().cpu(), targets.detach().cpu())
                             save_representative_batch(representative_batch)


                        log_first_2_batches_outputs_inputs_targets_predicted_mask(batch_idx,outputs,inputs,targets,predicted_mask,train_logger)


                        combined_loss, mask_loss, hybrid_loss = criterion(predicted_mask, inputs, targets)
              
            
                    print(f"[Batch] Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(combined_train_loader)}," f"Combined Loss: {combined_loss.item():.6f}, Mask Loss: {mask_loss.item():.6f}, Hybrid Loss: {hybrid_loss.item():.6f}")
          

                    Append_loss_values_for_batch(mask_loss, hybrid_loss, combined_loss)

                    
                    model_engine.backward(combined_loss)
                    model_engine.step()
                  


                    train_logger.info( f"[Batch] Epoch: {epoch + 1}/{epochs}, Batch: {batch_idx}/{len(combined_train_loader)}, " f"Combined Loss: {combined_loss.item():.6f}, Mask Loss: {mask_loss.item():.6f}, Hybrid Loss: {hybrid_loss.item():.6f}"  )
                    print( f"[Batch] Epoch: {epoch + 1}/{epochs}, Batch: {batch_idx}/{len(combined_train_loader)}, "    f"Combined Loss: {combined_loss.item():.6f}, Mask Loss: {mask_loss.item():.6f}, Hybrid Loss: {hybrid_loss.item():.6f}" )



                    # Check for NaN/Inf in loss
                    if torch.isnan(combined_loss) or torch.isinf(combined_loss):
                        train_logger.error(f"Skipping Batch {batch_idx} due to NaN/Inf in loss.")
                        continue

                    # Check for invalid outputs
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        train_logger.error(f"Skipping Batch {batch_idx} due to NaN/Inf in outputs.")
                        continue

                    running_loss += combined_loss.item()

      

                
            
                train_logger.debug(f"Loss History Lengths: Mask Loss: {len(loss_history_Epoches['mask_loss'])}, "    f"Hybrid Loss: {len(loss_history_Epoches['hybrid_loss'])}, " f"Combined Loss: {len(loss_history_Epoches['combined'])}, "   f"Total Loss: {len(loss_history_Epoches['Total_loss_per_epoch'])}")
               

               
                # Current Learning Rate logging
                current_lr = optimizer.param_groups[0]['lr']
                train_logger.info(f"Current Learning Rate {current_lr}")



                avg_epoch_loss = running_loss / len(combined_train_loader)
                train_logger.info(f"[Epoch {epoch + 1}] Average Training Loss: {avg_epoch_loss:.6f}")
                print(f"[Epoch {epoch + 1}] Average Training Loss: {avg_epoch_loss:.6f}")



                prev_epoch_loss_log(train_logger, prev_epoch_loss, avg_epoch_loss, epoch)



                if avg_epoch_loss < best_loss:
                    print("New best epoch loss! saving model now")
                    best_loss = avg_epoch_loss
                    train_logger.info(f"Avg_epoch_loss={avg_epoch_loss}, bestloss= {best_loss}")
                    checkpoint_dir = os.path.join(Model_CheckPoint, f"checkpoint_epochsss_{epoch + 1}")
                    try:
                       model_engine.save_checkpoint(checkpoint_dir)
                       print("model saved successfully ")
                    except Exception as e:
                        print(f"Error while saving checkpoint: {e}")
                        train_logger.error(f"Error while saving checkpoint: {e}")
                    print(f"[Epoch {epoch + 1}] Saving checkpoint to {checkpoint_dir} with loss {avg_epoch_loss:.6f}")
                    train_logger.info(f"[Epoch {epoch + 1}] Saving checkpoint to {checkpoint_dir} with loss {avg_epoch_loss:.6f}")
                else:
                    print(f"NO new best model. Previous best epoch loss: {best_loss:.6f}, This epoch's loss: {avg_epoch_loss:.6f}")
                    trigger_times += 1
                    train_logger.info(f"Trigger times: {trigger_times}/{patience}")              
                    print(f"Trigger times: {trigger_times}/{patience}")              
                if Early_break(trigger_times, patience, train_logger):
                    print(f"[Epoch {epoch + 1}] Early stopping triggered.")
                    break



                # Calculate average losses for epoch
                maskloss_avg, hybridloss_avg, combined_loss_avg = Get_calculated_average_loss_from_batches()

     

                # Logging memory after each epoch
                log_memory_after_index_epoch(epoch)

                # Logging average loss per epoch
                logging_avg_loss_epoch(epoch, prev_epoch_loss, epochs, avg_epoch_loss, maskloss_avg, hybridloss_avg, train_logger)

                prev_epoch_loss = avg_epoch_loss
                print(f"Prev_epoch_loss={prev_epoch_loss}, avg_epoch_loss= {avg_epoch_loss}")

                print("Appending loss values for epoches.")
                Append_loss_values_for_epoches(maskloss_avg, prev_epoch_loss, epochs, hybridloss_avg, combined_loss_avg, avg_epoch_loss)

            # Create loss diagrams after training
            create_loss_diagrams(loss_history_Batches, loss_history_Epoches)      

            # Training completion
            training_completed()
            print("[Training Complete] Finalizing training and saving model.")
            
            # Save the best and final models
            save_best_model(model_engine, best_model_path, Final_model_path, train_logger)
            print("saved best model")


            # Fine-tuning if flag is set
            if fine_tuning_flag:
                Automatic_Fine_Tune(
                    combined_val_loader,
                    combined_train_loader,
                    fine_tuned_model_base_path,
                    Final_model_path,
                    best_model_path
                )

            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

        except Exception as e:
            train_logger.error(f"[Train] Error during training: {str(e)}")
            print(f"[Train] Error during training: {str(e)}")
    else:
        train_logger.info("Training was not started.")
        print("Training was not started.")

if __name__ == "__main__":

    train()
    
    

