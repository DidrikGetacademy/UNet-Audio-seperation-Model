import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import torch
import os
import sys
import deepspeed
from torch import autocast
from tabulate import tabulate
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
#print("Current working directory:", os.getcwd())
#print("Current sys.path:", sys.path)
#print(f"project root {project_root}")
import json
from Training.Externals.Dataloader import create_dataloaders
from Training.Externals.Value_storage import  Append_loss_values_for_batch, Append_loss_values_for_epoches, Get_calculated_average_loss_from_batches, get_loss_value_list
from Training.Externals.Logger import setup_logger
from Model_Architecture.model import UNet,Model_Structure_Information
from Training.Externals.Loss_Class_Functions import Combinedloss 
from Training.Externals.Memory_debugging import log_memory_after_index_epoch,clear_memory_before_training
from Training.Externals.Loss_Diagram_Values import  create_loss_diagrams,create_loss_tabel_epoches,create_loss_table_batches
from Training.Externals.Functions import ( training_completed, load_model_path_func, save_best_model, Return_root_dir,)
from Training.Externals.Debugging_Values import check_Nan_Inf_loss, print_inputs_targets_shape,dataset_sample_information,logging_avg_loss_epoch,logging_avg_loss_epoch,log_first_2_batches_inputs_targets,log_first_2_batches_outputs_inputs_targets_predicted_mask,prev_epoch_loss_log
from Training.Validation import Validate_ModelEngine
root_dir = Return_root_dir() #Gets the root directory
fine_tuned_model_base_path = os.path.join(root_dir, "Model_weights/Fine_tuned")
Model_CheckPoint = os.path.join(root_dir, "Model_Weights/CheckPoints")
Final_model_path = os.path.join(root_dir, "Model_Weights/Pre_trained")
train_log_path = os.path.join(root_dir, "Model_performance_logg/log/Model_Training_logg.txt")
loss_log_path = os.path.join(root_dir,"Model_Performance_logg/log/loss_values.txt")
train_logger = setup_logger('train', train_log_path)
loss_logger = setup_logger("loss",loss_log_path)
with open(os.path.join(root_dir, "DeepSeed_Configuration/ds.config.json"), "r") as f:
    ds_config = json.load(f)



def train(start_training=True):
    load_model_path = os.path.join(Model_CheckPoint, "CheckPoints/checkpoint_epochsss_25")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_logger.info(f"[Train] Using device: {device}")
    epochs = 15
    prev_epoch_loss = None
    best_model_path = None 
    current_step = 0
    maskloss_avg, hybridloss_avg, combined_loss_avg = 0.0, 0.0, 0.0 
    num_workers = 0
    sampling_rate = 44100
    max_length_seconds = 15
    tried = 0

    
    clear_memory_before_training()

    #UNet - Model initialization, Optimizer Config, Custom Hybridloss function, Gradscaler.
    model = UNet(in_channels=1, out_channels=1).to(device)


    #initializing deepspeed
    model_engine, optimizer, _, _ = deepspeed.initialize(  model=model, model_parameters=model.parameters(), config_params=ds_config)

    if ds_config.get("fp16", {}).get("enabled", False):
       model_engine.to(dtype=torch.float16)

    #loading model if exists.
    load_model_path_func(load_model_path, model_engine, model, device)

    

    #loss functionality
    criterion = Combinedloss()


    #prininting Model Structure/Information
    Model_Structure_Information(model)

     

    if 'combined_train_loader' not in globals():
         train_loader, val_loader_phase = create_dataloaders(
            musdb18_dir=MUSDB18_dir,
            dsd100_dir=DSD100_dataset_dir,
            batch_size=ds_config["train_micro_batch_size_per_gpu"], 
            num_workers=num_workers,
            sampling_rate=sampling_rate,
            max_length_seconds=max_length_seconds,
            max_files_train=None,
            max_files_val=None,
    )
    if tried <= 1:
        dataset_sample_information(train_loader,val_loader_phase)
        tried += 1


#TRAINING LOOP STARTS HERE.....
    if start_training:
        try:
            for epoch in range(epochs):
                model_engine.train()
                running_loss = 0.0
                train_logger.info(f"[Train] Epoch {epoch + 1}/{epochs} started.\n")
                print(f"[Train] Epoch {epoch + 1}/{epochs} started.")

                for batch_idx, (inputs, targets) in enumerate(train_loader, start=1):
                    current_step += 1
                    inputs = inputs.to(device,  non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    #Prints sample of the dataset.
                    log_first_2_batches_inputs_targets(batch_idx,train_logger,inputs,targets)
                    print_inputs_targets_shape(inputs, targets, batch_idx)
              

                   ###AUTOCAST###
                    model_engine.zero_grad()
    
                    with autocast(device_type='cuda', enabled=(device.type == 'cuda'),dtype=torch.float16):
                        train_logger.debug(f"inputs into the --> model {inputs.shape}\n")
                        predicted_mask, outputs = model_engine(inputs)
                        predicted_vocals = predicted_mask * inputs
        
                        
                        log_first_2_batches_outputs_inputs_targets_predicted_mask(batch_idx, outputs, inputs, targets, predicted_mask, train_logger,predicted_vocals)

                        combined_loss, mask_loss, hybrid_loss_val, l1_loss_val, stft_loss_val  = criterion(predicted_mask, inputs, targets)

        
                    check_Nan_Inf_loss(combined_loss,batch_idx,outputs)
                    Append_loss_values_for_batch(mask_loss, hybrid_loss_val, combined_loss)

                    model_engine.backward(combined_loss)
                    model_engine.step()
                    running_loss += combined_loss.item()
                  
                    loss_logger.info(
                        f"[Batch] Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_loader)}," 
                        f"Combined Loss: {combined_loss.item():.6f},"
                        f"Mask Loss: {mask_loss.item():.6f},"  
                        f"Hybrid Loss: {hybrid_loss_val.item():.6f},"
                        f"L1 Loss: {l1_loss_val.item()}," 
                        f"STFT Loss: {stft_loss_val.item()},"
                        )



                 
    
            
        
               

               
                # Current Learning Rate logging
                current_lr = optimizer.param_groups[0]['lr']
                train_logger.info(f"Current Learning Rate {current_lr}\n")


                avg_epoch_loss = running_loss / len(train_loader)
                
                

                Validate_ModelEngine(epoch,model_engine,val_loader_phase,criterion,Model_CheckPoint, current_step)

               
                prev_epoch_loss_log(train_logger, prev_epoch_loss, avg_epoch_loss, epoch)

                # Calculate average losses for epoch
                maskloss_avg, hybridloss_avg, combined_loss_avg = Get_calculated_average_loss_from_batches()

     
                #Logging memory after each epoch
                log_memory_after_index_epoch(epoch)

                #Logging average loss per epoch
                logging_avg_loss_epoch(epoch, prev_epoch_loss, epochs, avg_epoch_loss, maskloss_avg, hybridloss_avg, train_logger)

                prev_epoch_loss = avg_epoch_loss
                
                
                loss_logger.info(
                    f"[Epoch {epoch + 1}]"
                    f"Previous Epoch Loss: {prev_epoch_loss}"
                    f"Average Training Loss: {avg_epoch_loss:.6f}\n"
                    )
           
                loss_logger.info(
                    f"Appending loss values for epoches, [AVERAGES] -->"
                    f"maskloss_avg={maskloss_avg}"
                    f"hybridloss_avg={hybridloss_avg}"
                    f"combined_loss_avg={combined_loss_avg}"
                    f"avg_epoch_loss={avg_epoch_loss}"
                    )
                
                
                Append_loss_values_for_epoches(maskloss_avg, hybridloss_avg, combined_loss_avg, avg_epoch_loss,loss_logger)


            # Create loss diagrams after training
            loss_history_Batches, loss_history_Epoches = get_loss_value_list(loss_logger)

            loss_logger.info(
                f"Loss_history_batches = {len(loss_history_Batches)}" 
                f"loss_history_epoches = {len(loss_history_Epoches)}"
                )
            
            
            create_loss_diagrams(loss_history_Batches, loss_history_Epoches,loss_logger)      
            create_loss_tabel_epoches(loss_history_Epoches,loss_logger)
            create_loss_table_batches(loss_history_Batches,loss_logger)

 
            training_completed()
            

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
    
    

