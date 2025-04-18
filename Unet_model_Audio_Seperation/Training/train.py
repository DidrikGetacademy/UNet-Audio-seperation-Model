import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import torch
import os
import sys
import gc
import deepspeed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
import json
from Training.Externals.Functions import Early_break
from Training.Externals.Dataloader import create_dataloader_training
from Training.Externals.Value_storage import  Append_loss_values_for_batch, Append_loss_values_for_epoches, Get_calculated_average_loss_from_batches, get_loss_value_list
from Training.Externals.Logger import setup_logger
from Model_Architecture.model import UNet,Model_Structure_Information
from Training.Externals.Loss_Class_Functions import Combinedloss 
from Training.Externals.Memory_debugging import log_memory_after_index_epoch,clear_memory_before_training
from Training.Externals.Loss_Diagram_Values import  create_loss_diagrams,create_loss_table_epoches,create_loss_table_batches
from Training.Externals.Functions import ( training_completed, load_model_path_func, save_best_model,save_checkpoint)
from Training.Externals.Debugging_Values import check_Nan_Inf_loss, print_inputs_targets_shape,dataset_sample_information,logging_avg_loss_epoch,logging_avg_loss_epoch,log_first_2_batches_inputs_targets,log_first_2_batches_outputs_inputs_targets_predicted_mask,prev_epoch_loss_log
from Training.Validation import Validate_ModelEngine
from Training.Externals.utils import Return_root_dir
from Training.Fine_Tuned_model import fine_tune_model
from Training.Evaluation import run_evaluation
from Datasets.Scripts.Dataset_utils import Convert_spectrogram_to_audio
root_dir = Return_root_dir()
fine_tuned_model_base_path = os.path.join(root_dir, "Model_weights/Fine_tuned")
Model_CheckPoint_Training = os.path.join(root_dir, "Model_Weights/CheckPoints/Training")
Model_CheckPoint_Evaluation = os.path.join(root_dir, "Model_Weights/CheckPoints/Evaluation")
Final_model_path = os.path.join(root_dir, "Model_Weights/Pre_trained")
train_log_path = os.path.join(root_dir, "Model_performance_logg/log/Model_Training_logg.txt")
loss_log_path = os.path.join(root_dir,"Model_Performance_logg/log/loss_values.txt")
eval_log_path = os.path.join(root_dir, "Model_Performance_logg/log/Evaluation_logg.txt")
train_logger = setup_logger('train', train_log_path)
loss_logger = setup_logger("loss",loss_log_path)
Evaluation_logger = setup_logger('Evaluation', eval_log_path)
validation_log_path = os.path.join(root_dir, "Model_Performance_logg/log/Validation_logg.txt")
Validation_logger = setup_logger('Validation', validation_log_path)
checkpoint_path = os.path.join(root_dir, "Model_Weights/Best_model.pth") 
data_loader_path = os.path.join(root_dir, "Model_Performance_logg/log/Dataloader.txt")
data_loader = setup_logger('dataloader', data_loader_path)
trigger_times = 0
os.environ['MASTER_PORT'] = '29501'  

with open(os.path.join(root_dir, "DeepSeed_Configuration/ds_config_Training.json"), "r") as f:
    ds_config = json.load(f)

def train(start_training=True):
    clear_memory_before_training()
    model_path_temp="/mnt/c/Users/didri/Desktop/Programmering/ArtificalintelligenceModels/UNet-Model_Vocal_Isolation/Unet_model_Audio_Seperation/Model_Weights/CheckPoints/Evaluation/checkpoint_epoch_3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_logger.info(f"[Train] Using device: {device}")
    maskloss_avg, hybridloss_avg, combined_loss_avg = 0.0, 0.0, 0.0 
    current_step = 0
    epochs = 10
    patience = 12
    bestloss = float('inf')
    bestloss_validation = float('inf')
    prev_epoch_loss = float('inf')
    total_batches_processed = 0  
    Dataset_count_test = float('inf')
    model = UNet(in_channels=1, out_channels=1).to(device)

    for name, param in model.named_parameters():
       print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")

    Model_Structure_Information(model)

    model_engine, optimizer, _, _ = deepspeed.initialize(
         
        model=model,
        model_parameters=model.parameters(),
        config_params=ds_config
    )

    load_model_path_func(model_path_temp, model_engine, model, device)

    criterion = Combinedloss()

    if 'combined_train_loader' not in globals():
         train_loader, val_loader_phase = create_dataloader_training(
            batch_size=ds_config["train_micro_batch_size_per_gpu"], 
            num_workers=6,
            max_length_seconds=11,
            max_files_train=150,
            max_files_val=30,

    )
         
    if Dataset_count_test <= 1:
        dataset_sample_information(train_loader,val_loader_phase)
        Dataset_count_test += 1

#TRAINING LOOP STARTS HERE.....<
    if start_training:
        try:
            for epoch in range(epochs):

                model_engine.train()
                running_loss = 0.0
                train_logger.info(f"[Train] Epoch {epoch + 1}/{epochs} started.\n")
                print(f"[Train] Epoch {epoch + 1}/{epochs} started.")

                for batch_idx, (inputs, targets) in enumerate(train_loader, start=1):
                    print(f"[EPOCH {epoch + 1}]Starting... Batch: {batch_idx}/{len(train_loader)} inputs:{inputs.shape}, targets: {targets.shape}")
                    if inputs.numel() == 0 or targets.numel() == 0:
                        data_loader.warning(f"[EPOCH {epoch + 1}]Skipping batch {batch_idx} because inputs or targets are empty.")
                        continue  

                    
                    current_step += 1


                    inputs = inputs.to(device, dtype=torch.float16, non_blocking=True)
                    targets = targets.to(device, dtype=torch.float16, non_blocking=True)

           
                    log_first_2_batches_inputs_targets(batch_idx,train_logger,inputs,targets)
                    print_inputs_targets_shape(inputs, targets, batch_idx)
              

                    model_engine.zero_grad()    
                    train_logger.debug(f"inputs into the --> model {inputs.shape}\n")
                    predicted_mask, outputs = model_engine(inputs)
                     

                    predicted_vocals = predicted_mask * inputs
    
                    if batch_idx <= 2:
                           train_logger.info(f"[EPOCH {epoch + 1}]outputs from the model {outputs.shape}\nfor batch {batch_idx}\npredicted mask shape {predicted_mask.shape}")
                           Convert_spectrogram_to_audio(audio_path="/mnt/c/Users/didri/Desktop/Programmering/ArtificalintelligenceModels/UNet-Model_Vocal_Isolation/Unet_model_Audio_Seperation/audio_logs/Trening",predicted_vocals=predicted_vocals[0],targets=targets[0],inputs=inputs[0],outputs=None)
                           


                    log_first_2_batches_outputs_inputs_targets_predicted_mask(batch_idx, outputs, inputs, targets, predicted_mask, train_logger,predicted_vocals)

                    combined_loss, mask_loss, hybrid_loss_val, l1_loss_val, stft_loss_val, sdr_loss = criterion(predicted_mask, inputs, targets)
                    total_batches_processed += 1
                    data_loader.info(f"\n[EPOCH {epoch + 1}] Currently Total batches processed during this training is: [{total_batches_processed}] out of [{len(train_loader)}]\n")

        
                    check_Nan_Inf_loss(combined_loss,batch_idx,outputs)
                    Append_loss_values_for_batch(mask_loss, hybrid_loss_val, combined_loss)

                    model_engine.backward(combined_loss)
                    model_engine.step()
                    running_loss += combined_loss.item()
                  
                    loss_logger.info(
                        f"[EPOCH {epoch + 1}][Batch] Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_loader)},\n" 
                        f"Combined Loss: {combined_loss.item():.6f}\n"
                        f"Mask Loss: {mask_loss.item():.6f}\n"  
                        f"Hybrid Loss: {hybrid_loss_val.item():.6f}\n"
                        f"L1 Loss: {l1_loss_val.item()}\n" 
                        f"STFT Loss: {stft_loss_val.item()}\n"
                        f"SDR Loss: {sdr_loss.item()}\n"
                        )

         
                current_lr = optimizer.param_groups[0]['lr']
                train_logger.info(f"Current Learning Rate {current_lr}\n")


                avg_epoch_loss = running_loss / len(train_loader)
                if avg_epoch_loss < bestloss:
                        global trigger_times
                        bestloss  = avg_epoch_loss
                        save_checkpoint(model_engine, optimizer, epoch, avg_epoch_loss, Model_CheckPoint_Training)
                        trigger_times = 0
                        train_logger.info(f"\n [EPOCH {epoch + 1}]Model Checkpoint SAVED at epoch {epoch + 1}\n avg_epoch_loss: [{avg_epoch_loss:.6f}]\n bestloss: [{bestloss:.6f}]\n Trigger times: {trigger_times}/{patience}\n")
                else: 
                    trigger_times += 1
                    train_logger.info(f"\n[EPOCH {epoch + 1}]No improvement, NO NEW MODEL CHECKPOINT SAVED at epoch {epoch + 1}\n avg_epoch_loss: [{avg_epoch_loss:.6f}]\n bestloss: [{bestloss:.6f}]\n Trigger: {trigger_times}/{patience}\n")

                
    
                bestloss_validation, trigger_times, avg_validation_loss = Validate_ModelEngine(epoch,model_engine,val_loader_phase,criterion,Model_CheckPoint_Evaluation, current_step)
                Validation_logger.info(f"\n[EPOCH {epoch + 1}]Validate ----> [bestloss_validation]: {bestloss_validation:.6f},]\n [trigger_times: {trigger_times}],\n [avg_validation_loss: {avg_validation_loss:.6f}]\n")

                if Early_break(trigger_times, patience):
                    train_logger.warning(f"[EPOCH {epoch + 1}]Early stopping triggered at epoch {epoch + 1}")
                    break


    
                prev_epoch_loss_log(train_logger, prev_epoch_loss, avg_epoch_loss, epoch)

        
                maskloss_avg, hybridloss_avg, combined_loss_avg = Get_calculated_average_loss_from_batches(loss_logger)

        
                log_memory_after_index_epoch(epoch)
         
            
                logging_avg_loss_epoch(epoch, prev_epoch_loss, epochs, avg_epoch_loss, maskloss_avg, hybridloss_avg, train_logger)

                prev_epoch_loss = avg_epoch_loss
     
                
                loss_logger.info(
                    f"\n[Epoch {epoch + 1}]\n"
                    f"Previous Epoch Loss: {prev_epoch_loss}\n"
                    f"Average Training Loss: {avg_epoch_loss:.6f}\n"
                    )
           
                loss_logger.info(
                    f"\nAppending loss values for epoches, [AVERAGES] -->\n"
                    f"maskloss_avg={maskloss_avg}\n"
                    f"hybridloss_avg={hybridloss_avg}\n"
                    f"combined_loss_avg={combined_loss_avg}\n"
                    f"avg_epoch_loss={avg_epoch_loss}\n"
                    )
                
         
                Append_loss_values_for_epoches(maskloss_avg, hybridloss_avg, combined_loss_avg, avg_epoch_loss,loss_logger)

                train_logger.info(f"Epoch: {epoch  + 1} COMPLETED\n\n\n\n\n")


            loss_history_Batches, loss_history_Epoches = get_loss_value_list()

            loss_logger.info(
                f"Loss_history_batches = {len(loss_history_Batches)}\n" 
                f"loss_history_epoches = {len(loss_history_Epoches)}\n"
                )
            
            create_loss_diagrams(loss_history_Batches, loss_history_Epoches,loss_logger)      

            create_loss_table_epoches(loss_history_Epoches,loss_logger)

            create_loss_table_batches(loss_history_Batches,loss_logger)
 
            training_completed()

            save_best_model(model_engine, bestloss, Final_model_path)



            #Evaluation
            try:
                train_logger.info("Starting Evaluation now. ")
                avg_loss, avg_sdr, avg_sir, avg_sar =  run_evaluation(model_engine=model_engine, device=device, checkpoint_path=Final_model_path)
                Evaluation_logger.info(f"[BEFORE FINE TUNING] average_loss: {avg_loss:.6f}, average_sdr: {avg_sdr:.4f}, average_sir: {avg_sir:.4f}, average_sar: {avg_sar:.4f}")
            except Exception as e:
                print(f"Error during evaluation: {str(e)}")




          #Fine-Tuning
          #  try:
          #     final_model_path = os.path.join(Final_model_path, "final_model_best_model.pth")
          #     fine_tuned_model_path = os.path.join(root_dir,"Model_Weights/Fine_Tuned/Model.pth")
          #     fine_tune_model(fine_tuned_model_path=fine_tuned_model_path,  Fine_tuned_training_loader=train_loader,  Finetuned_validation_loader = val_loader_phase,  model_engine=model_engine,  fine_tune_epochs=10,  pretrained_model_path=final_model_path)
          #  except Exception as e:
          #        print(f"Error during Fine_tuning: {str(e)} ") 
   

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

