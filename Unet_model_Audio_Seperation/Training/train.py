import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import torch
import os
import sys
import gc
import deepspeed
import gc
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
from Training.Externals.Loss_Diagram_Values import  create_loss_diagrams,create_loss_tabel_epoches,create_loss_table_batches
from Training.Externals.Functions import ( training_completed, load_model_path_func, save_best_model)
from Training.Externals.Debugging_Values import check_Nan_Inf_loss, print_inputs_targets_shape,dataset_sample_information,logging_avg_loss_epoch,logging_avg_loss_epoch,log_first_2_batches_inputs_targets,log_first_2_batches_outputs_inputs_targets_predicted_mask,prev_epoch_loss_log,save_audio_files_from_model_dataset_mask
from Training.Validation import Validate_ModelEngine
from Training.Externals.utils import Return_root_dir
from Training.Fine_Tuned_model import fine_tune_model
from Training.Validation import Validate_ModelEngine
root_dir = Return_root_dir()
fine_tuned_model_base_path = os.path.join(root_dir, "Model_weights/Fine_tuned")
Model_CheckPoint_Training = os.path.join(root_dir, "Model_Weights/CheckPoints/Training")
Model_CheckPoint_Evaluation = os.path.join(root_dir, "Model_Weights/CheckPoints/Evaluation")
Final_model_path = os.path.join(root_dir, "Model_Weights/Pre_trained")
train_log_path = os.path.join(root_dir, "Model_performance_logg/log/Model_Training_logg.txt")
loss_log_path = os.path.join(root_dir,"Model_Performance_logg/log/loss_values.txt")
train_logger = setup_logger('train', train_log_path)
loss_logger = setup_logger("loss",loss_log_path)



with open(os.path.join(root_dir, "DeepSeed_Configuration/ds_config_Training.json"), "r") as f:
    ds_config = json.load(f)



def train(start_training=True):
    clear_memory_before_training()
    model_path_temp=r'/mnt/c/Users/didri/Desktop/Programmering/ArtificalintelligenceModels/UNet-Model_Vocal_Isolation/Unet_model_Audio_Seperation/Model_Weights/CheckPoints/Training/checkpoint_epoch_1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_logger.info(f"[Train] Using device: {device}")
    maskloss_avg, hybridloss_avg, combined_loss_avg = 0.0, 0.0, 0.0 
    epochs = 5
    current_step = 0
    patience = 50
    trigger_times = 0
    prev_epoch_loss = None
    best_model_path = None 
    tried = float('inf')
    bestloss = float('inf')



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
            num_workers=8,
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
                    print(f"Starting... Batch: {batch_idx} inputs:{inputs.shape}, targets: {targets.shape}")
                    current_step += 1


                    inputs = inputs.to(device, dtype=torch.float16, non_blocking=True)
                    targets = targets.to(device, dtype=torch.float16, non_blocking=True)

           
                    log_first_2_batches_inputs_targets(batch_idx,train_logger,inputs,targets)
                    print_inputs_targets_shape(inputs, targets, batch_idx)
              

                    model_engine.zero_grad()
    
                    with torch.amp.autocast(device_type='cuda',dtype=torch.float16):
                        train_logger.debug(f"inputs into the --> model {inputs.shape}\n")
                        predicted_mask, outputs = model_engine(inputs)
                        predicted_vocals = predicted_mask * inputs
    
                        if batch_idx <= 2:
                            try:
                                save_audio_files_from_model_dataset_mask(predicted_vocals, targets ,inputs, outputs)
                            except Exception as e:
                                     print(f"error durring reconstruction of audio: {str(e)}")


                        log_first_2_batches_outputs_inputs_targets_predicted_mask(batch_idx, outputs, inputs, targets, predicted_mask, train_logger,predicted_vocals)


                        combined_loss, mask_loss, hybrid_loss_val, l1_loss_val, stft_loss_val = criterion(predicted_mask, inputs, targets)


        
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
                    

               

         
                current_lr = optimizer.param_groups[0]['lr']
                train_logger.info(f"Current Learning Rate {current_lr}\n")


                avg_epoch_loss = running_loss / len(train_loader)
                
                

                Validate_ModelEngine(epoch,model_engine,val_loader_phase,criterion,Model_CheckPoint_Evaluation, current_step)
                gc.collect()
                torch.cuda.empty_cache()
                if running_loss < bestloss:
                    train_logger.info(f"New best model (Loss: {running_loss:.6f} < {bestloss:.6f})")
                    bestloss = running_loss
                    checkpoint_dir = os.path.join(Model_CheckPoint_Training, f"checkpoint_epoch_{epoch + 1}")
                    try:
                        model_engine.save_checkpoint(checkpoint_dir, {
                            'step': current_step,
                            'best_loss': bestloss,
                            'trigger_times': trigger_times
                        })
                        trigger_times = 0
                    except Exception as e:
                            train_logger.error(f"Checkpoint save failed: {str(e)}")
                else:
                    trigger_times += 1
                    train_logger.info(f"No improvement (Trigger: {trigger_times}/{patience})")

            
                if Early_break(trigger_times, patience):
                    train_logger.warning(f"Early stopping triggered at epoch {epoch + 1}")

                    return bestloss, trigger_times, running_loss

               
                prev_epoch_loss_log(train_logger, prev_epoch_loss, avg_epoch_loss, epoch)

             
                maskloss_avg, hybridloss_avg, combined_loss_avg = Get_calculated_average_loss_from_batches(loss_logger)

     
            
                log_memory_after_index_epoch(epoch)

            
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


            
            loss_history_Batches, loss_history_Epoches = get_loss_value_list()

            loss_logger.info(
                f"Loss_history_batches = {len(loss_history_Batches)}\n" 
                f"loss_history_epoches = {len(loss_history_Epoches)}\n"
                )
            
            
            create_loss_diagrams(loss_history_Batches, loss_history_Epoches,loss_logger)      
            create_loss_tabel_epoches(loss_history_Epoches,loss_logger)
            create_loss_table_batches(loss_history_Batches,loss_logger)

 
            training_completed()

            save_best_model(model_engine, best_model_path, Final_model_path)
            print("Starting Validation on model performance after training...")
           
            avg_loss_before_finetune, avg_sdr_after_finetune, avg_sir_before_finetune, avg_sar_before_finetune = Validate_ModelEngine(epoch,model_engine, val_loader_phase,criterion,Model_CheckPoint_Evaluation,current_step)
            final_model_path = os.path.join(Final_model_path, "final_model_best_model.pth")
            fine_tuned_model_path = os.path.join(root_dir,"Model_Weights/Fine_Tuned/Model.pth")
            print("Starting fine tuning After validation....")
            loss_logger.info(f"Before Fine-tuning - Average Loss: {avg_loss_before_finetune:.6f}")
            loss_logger.info(f"Before Fine-tuning - Average SDR: {avg_sdr_after_finetune:.4f}, SIR: {avg_sir_before_finetune:.4f}, SAR: {avg_sar_before_finetune:.4f}")

            #fine_tune_model(fine_tuned_model_path=fine_tuned_model_path,  Fine_tuned_training_loader=train_loader,  Finetuned_validation_loader = val_loader_phase,  ds_config=ds_config,  fine_tune_epochs=10,  pretrained_model_path=final_model_path)



            avg_loss_after_finetune, avg_sdr_after_finetune, avg_sir_after_finetune, avg_sar_after_finetune = Validate_ModelEngine(model_engine, val_loader_phase, device)
            loss_logger.info(f"After Fine-tuning - Average Loss: {avg_loss_after_finetune:.6f}")
            loss_logger.info(f"After Fine-tuning - Average SDR: {avg_sdr_after_finetune:.4f}, SIR: {avg_sir_after_finetune:.4f}, SAR: {avg_sar_after_finetune:.4f}")
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
    
    

