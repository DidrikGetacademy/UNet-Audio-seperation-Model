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
import json
from Training.Externals.Dataloader import create_dataloaders
from Training.Externals.Value_storage import  Append_loss_values_for_batch, Append_loss_values_for_epoches, Get_calculated_average_loss_from_batches, loss_history_Epoches,loss_history_Batches
from Training.Externals.Logger import setup_logger,logging_avg_loss_epoch,prev_epoch_loss_log,log_first_2_batches_inputs_targets,log_first_2_batches_outputs_inputs_targets_predicted_mask
from Model_Architecture.model import UNet,Model_Structure_Information
from Training.Externals.Loss_Class_Functions import Combinedloss 
from Training.Externals.Memory_debugging import log_memory_after_index_epoch,clear_memory_before_training
from Training.Externals.Loss_Diagram_Values import  create_loss_diagrams
import logging
from Training.Externals.Functions import ( training_completed, load_model_path_func, Early_break, save_best_model, Return_root_dir,)
from Training.Externals.Debugging_Values import check_inputs_targets_dataset, print_inputs_targets_shape,dataset_sample_information
#PATHS
root_dir = Return_root_dir() #Gets the root directory
fine_tuned_model_base_path = os.path.join(root_dir, "Model_weights/Fine_tuned")
Model_CheckPoint = os.path.join(root_dir, "Model_Weights/CheckPoints")
Final_model_path = os.path.join(root_dir, "Model_Weights/Pre_trained")
MUSDB18_dir = os.path.join(root_dir, "Datasets/Dataset_Audio_Folders/musdb18")
DSD100_dataset_dir = os.path.join(root_dir, "Datasets/Dataset_Audio_Folders/DSD100")
custom_dataset_dir = os.path.join(root_dir,"Datasets/Dataset_Audio_Folders/Custom_Dataset")
train_log_path = os.path.join(root_dir, "Model_performance_logg/log/Model_Training_logg.txt")
train_logger = setup_logger('train', train_log_path,level=logging.INFO)
with open(os.path.join(root_dir, "Training/ds.config.json"), "r") as f:
    ds_config = json.load(f)






def train(start_training=True):
    load_model_path = os.path.join(Model_CheckPoint, "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_logger.info(f"[Train] Using device: {device}")
    epochs = 25
    patience = 5
    best_loss = float('inf')
    trigger_times = 0
    prev_epoch_loss = None
    best_model_path = None 
    current_step = 0
    maskloss_avg, hybridloss_avg, combined_loss_avg = 0.0, 0.0, 0.0 
    num_workers = 4
    sampling_rate = 44100
    max_length_seconds = 10



    
    clear_memory_before_training()

    #UNet - Model initialization, Optimizer Config, Custom Hybridloss function, Gradscaler.
    model = UNet(in_channels=1, out_channels=1).to(device)



    #initializing deepspeed
    model_engine, optimizer, _, _ = deepspeed.initialize(  model=model, model_parameters=model.parameters(), config_params=ds_config)


    #loading model if exists.
    load_model_path_func(load_model_path, model_engine, model, device)

    

    #loss functionality
    criterion = Combinedloss()


    #prininting Model Structure/Information
    Model_Structure_Information(model)

     

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
            max_files_CustomDataset=None,
    )



#TRAINING LOOP STARTS HERE.....
    if start_training:
        try:
            for epoch in range(epochs):
                model_engine.train()
                running_loss = 0.0
                train_logger.info(f"[Train] Epoch {epoch + 1}/{epochs} started.")
                print(f"[Train] Epoch {epoch + 1}/{epochs} started.")



                for batch_idx, (inputs, targets) in enumerate(combined_train_loader, start=1):
       
                    current_step += 1
                    
                    #Prints sample of the dataset.
                    dataset_sample_information(combined_train_loader,combined_val_loader)

                    #Moves the data to device. 
                    print(f"[Train] Moving batch {batch_idx} data to {device}...")
                    inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
                    targets = targets.to(device, dtype=torch.float32, non_blocking=True)


                    if batch_idx <= 2:
                        train_logger.info(f"[INPUTS MIN/MAX & TARGET MIN/MAX LOGGGING] --- Sample {batch_idx}: input min={inputs.min()}, max={inputs.max()}, target min={targets.min()}, max={targets.max()}")
                    train_logger.info(f"[Batch {batch_idx}] Batch size: {inputs.size(0)}")




                    # Logging and Data checking!
                    log_first_2_batches_inputs_targets(batch_idx,train_logger,inputs,targets)
                    check_inputs_targets_dataset(inputs, targets, batch_idx)
                    print_inputs_targets_shape(inputs, targets, batch_idx)



                   ###AUTOCAST###
                    model_engine.zero_grad()
                    with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                        train_logger.debug(f"inputs --> model {inputs.shape}")
                        predicted_mask, outputs = model_engine(inputs)
                        predicted_vocals = predicted_mask * inputs
        
                        train_logger.debug(f"[Before Mask Application] Input min: {inputs.min()}, max: {inputs.max()}")
                        train_logger.debug(f"[Mask] Predicted mask min: {predicted_mask.min()}, max: {predicted_mask.max()}")
                        train_logger.debug(f"[After Mask Application] Predicted vocals min: {predicted_vocals.min()}, max: {predicted_vocals.max()}")

                    

                        log_first_2_batches_outputs_inputs_targets_predicted_mask(batch_idx,outputs,inputs,targets,predicted_mask,train_logger)
       

                        combined_loss, mask_loss, hybrid_loss = criterion(predicted_mask, inputs, targets)
                        train_logger.debug(f"[Loss Debugging] Combined Loss: {combined_loss.item()}, Mask Loss: {mask_loss.item()}, Hybrid Loss: {hybrid_loss.item()}")

            
                    train_logger.info(f"[Batch] Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(combined_train_loader)}," f"Combined Loss: {combined_loss.item():.6f}, Mask Loss: {mask_loss.item():.6f}, Hybrid Loss: {hybrid_loss.item():.6f}")
          

                    Append_loss_values_for_batch(mask_loss, hybrid_loss, combined_loss)

                    
                    model_engine.backward(combined_loss)
                    model_engine.step()
                    running_loss += combined_loss.item()
                  


                    train_logger.info( f"[Batch] Epoch: {epoch + 1}/{epochs}, Batch: {batch_idx}/{len(combined_train_loader)}, " f"Combined Loss: {combined_loss.item():.6f}, Mask Loss: {mask_loss.item():.6f}, Hybrid Loss: {hybrid_loss.item():.6f}"  )



                    #check for NaN/Inf in loss
                    if torch.isnan(combined_loss) or torch.isinf(combined_loss):
                        train_logger.error(f"Skipping Batch {batch_idx} due to NaN/Inf in loss.")
                        continue



                    #check for invalid outputs
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        train_logger.error(f"Skipping Batch {batch_idx} due to NaN/Inf in outputs.")
                        continue


      

                
            
                train_logger.debug(f"Loss History Lengths: Mask Loss: {len(loss_history_Epoches['mask_loss'])}, " f"Hybrid Loss: {len(loss_history_Epoches['hybrid_loss'])}, " f"Combined Loss: {len(loss_history_Epoches['combined'])}, "   f"Total Loss: {len(loss_history_Epoches['Total_loss_per_epoch'])}")
               

               
                # Current Learning Rate logging
                current_lr = optimizer.param_groups[0]['lr']
                train_logger.info(f"Current Learning Rate {current_lr}")



                avg_epoch_loss = running_loss / len(combined_train_loader)
                train_logger.info(f"[Epoch {epoch + 1}] Average Training Loss: {avg_epoch_loss:.6f}")



                prev_epoch_loss_log(train_logger, prev_epoch_loss, avg_epoch_loss, epoch)



                if avg_epoch_loss < best_loss:
                    print("New best epoch loss! saving model now")
                    best_loss = avg_epoch_loss
                    train_logger.info(f"Avg_epoch_loss={avg_epoch_loss}, bestloss= {best_loss}")
                    checkpoint_dir = os.path.join(Model_CheckPoint, f"checkpoint_epochsss_{epoch + 1}")
                    client_sd = {'step': current_step, 'best_loss': best_loss, 'trigger_times': trigger_times}           
                    try:
                       model_engine.save_checkpoint(checkpoint_dir,client_sd)
                       train_logger.info(f"New best model saved to {checkpoint_dir}, [Epoch {epoch + 1}] with client_sd: {client_sd} with loss {avg_epoch_loss:.6f}")
                    except Exception as e:
                        train_logger.error(f"Error while saving checkpoint: {e}")
                else:
                    trigger_times += 1
                    train_logger.info(f"NO IMPROVEMENT IN LOSS!. Previous best epoch loss: {best_loss:.6f}, This epoch's loss: {avg_epoch_loss:.6f}, trigger times: {trigger_times}")                      



                if Early_break(trigger_times, patience):
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

                train_logger.info(f"Appending loss values for epoches, values --> maskloss_avg={maskloss_avg}, hybridloss_avg={hybridloss_avg}, combined_loss_avg={combined_loss_avg}, avg_epoch_loss={avg_epoch_loss}")
                Append_loss_values_for_epoches(maskloss_avg, hybridloss_avg, combined_loss_avg, avg_epoch_loss)


            # Create loss diagrams after training
            create_loss_diagrams(loss_history_Batches, loss_history_Epoches)      

            # Training completion
            training_completed()
            
            # Save the best and final models
            save_best_model(model_engine, best_model_path, Final_model_path)

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
    
    

