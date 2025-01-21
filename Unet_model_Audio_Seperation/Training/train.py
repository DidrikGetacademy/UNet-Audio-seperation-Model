# train.py
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils as nn_utils
import os
import sys
import shutil
from torch import autocast, GradScaler
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Training.Externals.Dataloader import create_dataloaders
from Training.Fine_Tuned_model import fine_tune_model 
from Training.Externals.Logger import setup_logger
from Model_Architecture.model import UNet
from Training.Externals.Loss_Class_Functions import ( Combinedloss )

from Training.Externals.Memory_debugging import (
    clear_memory_before_training,
    log_memory_after_index_epoch,
    )

from Training.Externals.Loss_Diagram_Values import (
    log_spectrograms_to_tensorboard,
    create_loss_diagrams,
)

from Training.Externals.Functions import (
    Validate_epoch,
    save_final_model,
    training_completed,
    save_final_model,
    Model_Structure_Information,
    dataset_sample_information,
    check_inputs_targets_dataset,
    save_model_checkpoint,
    load_model_path_func,
    print_inputs_targets_shape,
)
from Training.Externals.Value_storage import (
    Append_loss_values_for_batch,
    Append_loss_values_for_epoches,
    Get_calculated_average_loss_from_batches,
)

from Training.Externals.Value_storage import (loss_history_Epoches,loss_history_Batches)



log_dir = r"C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_Performance_logg\Tensorboard"  
train_logger = setup_logger( 'train', r'C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_performance_logg\log\Model_Training_logg.txt')
Final_model_path = r"C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_Weights\Pre_trained"
fine_tuned_model_base_path = r"C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_weights\Fine_tuned"
Model_CheckPoint = r"C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_Weights\CheckPoints"
MUSDB18_dir = r'C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Datasets\Dataset_Audio_Folders\musdb18'
DSD100_dataset_dir =r'C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Datasets\Dataset_Audio_Folders\DSD100'
# Create necessary directories

os.makedirs(fine_tuned_model_base_path, exist_ok=True)
os.makedirs(Model_CheckPoint, exist_ok=True)
os.makedirs(Final_model_path, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)




def train(load_model_path=r"C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_Weights\Fine_tuned\model.pth",start_training=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_logger.info(f"[Train] Using device: {device}")
    clear_memory_before_training()

   #Training config
    batch_size = 4
    desired_effective_batch_size = 64
    accumulation_steps = desired_effective_batch_size // batch_size
    learning_rate = 1e-6
    epochs = 13
    patience = 2
    best_loss = float('inf')
    best_val_loss = float('inf')
    trigger_times = 0
    gradient_clip_value = 0.5 
    prev_epoch_loss = None
    best_model_path = None 
    maskloss_avg, hybridloss_avg, combined_loss_avg = 0.0, 0.0, 0.0 
 

    #UNet - Model initialization, Optimizer Config, Custom Hybridloss function, Gradscaler.
    model = UNet(in_channels=1, out_channels=1).to(device)

    # OPTIMIZER: Adam optimizer with customized settings
    optimizer = optim.Adam(
        model.parameters(),  # Parameters of the model to optimize
        lr=1e-3,             # Learning rate: controls the step size for weight updates
        weight_decay=1e-4,   # L2 regularization: penalizes large weights to reduce overfitting
        betas=(0.8, 0.9),    # Exponential decay rates for moving averages of gradients (momentum) and squared gradients
        eps=1e-6             # Small value added to denominator for numerical stability (useful for mixed precision)
    )

    # SCHEDULER: ReduceLROnPlateau to dynamically adjust learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,           # Optimizer instance to apply the scheduler to
        mode='min',          # Mode: 'min' indicates learning rate reduces when the metric stops decreasing
        factor=0.5,          # Multiplicative factor: reduces the learning rate by 50% when triggered
        patience=3,          # Number of epochs to wait for improvement before reducing LR
        threshold=1e-3,      # Minimum change in monitored metric to qualify as an improvement
        verbose=True,        # If True, logs messages about LR reductions
        cooldown=1,          # Number of epochs to wait after reducing LR before monitoring improvement again
        min_lr=1e-5          # Lower bound for the learning rate to prevent it from going too low
    )


    #SCALER
    scaler = GradScaler() #is used in mixed precision training to prevent numerical underflow when using float16 (FP16) precision. It scales the gradients during backpropagation to maintain numerical stability.


    #FULL LOSS 
    criterion = Combinedloss()

    #How many small batches.
    Effective_batch_size = batch_size * accumulation_steps


    #Prininting Model Structure/Information
    Model_Structure_Information(model)


    # Loading model if exists or creating a new one. 
    load_model_path_func(load_model_path, model, device)
     

    #Prints sample of the dataset.
    dataset_sample_information(combined_train_loader,combined_val_loader)


#TRAINING LOOP STARTS HERE.....

    if start_training:
        train_logger.info(f"Training started with configuration: Batch size={batch_size}, Effective batch size: {Effective_batch_size} Learning rate={learning_rate}, Epochs={epochs}")
        try:
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                representative_batch = None  
                train_logger.info(f"[Train] Epoch {epoch + 1}/{epochs} started.")

                #Resets gradients.
                optimizer.zero_grad()
                for batch_idx, (inputs, targets) in enumerate(combined_train_loader, start=1):

                    #Logs for the first 2 batches in each epoch.
                    if batch_idx < 2:    
                        train_logger.info(f"Batch {batch_idx}: Mixture shape={inputs.shape}, Target shape={targets.shape}")
                        train_logger.info(f"Mixture min={inputs.min().item():.4f}, max={inputs.max().item():.4f}")
                        train_logger.info(f"Target min={targets.min().item():.4f}, max={targets.max().item():.4f}")
                    
                    #Data check & logging.
                    check_inputs_targets_dataset(inputs, targets, batch_idx)
                    print_inputs_targets_shape(inputs, targets, batch_idx)

                    #Moves the data to device.
                    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

                   ###AUTOCAST###
                    with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                        predicted_mask, outputs = model(inputs.to(device,non_blocking=True))
                        predicted_vocals = predicted_mask * inputs

                        if batch_idx == 1:
                           representative_batch = (inputs.detach().cpu(), predicted_vocals.detach().cpu(), targets.detach().cpu())

                        if batch_idx < 2:
                            train_logger.info(f"Batch {batch_idx}: Mask range: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}")
                            train_logger.debug(f"Batch {batch_idx}: Inputs shape={inputs.shape}, Targets shape={targets.shape}, Predicted Mask shape={predicted_mask.shape}, Outputs shape={outputs.shape}")
                        
                        train_logger.debug(f"Mask min={predicted_mask.min().item()}, max={predicted_mask.max().item()}")
                        combined_loss, mask_loss, hybrid_loss = criterion(predicted_mask, inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True))




                        #TENSORBOARD 
                        global_step = epoch * len(combined_train_loader) + batch_idx
                        writer.add_scalar('Loss/Combined', combined_loss.item(), global_step)
                        writer.add_scalar('Loss/Mask', mask_loss.item(), global_step)
                        writer.add_scalar('Loss/Hybrid', hybrid_loss.item(), global_step)
               


                    #Normalizing the loss.
                    combined_loss = combined_loss / accumulation_steps




                    #Loss logging
                    train_logger.info(
                    f"[Batch] Epoch: {epoch + 1}/{epochs}, Batch: {batch_idx}/{len(combined_train_loader)}, "
                    f"Combined Loss: {combined_loss.item():.6f}, Mask Loss: {mask_loss.item():.6f}, Hybrid Loss: {hybrid_loss.item():.6f}"
                    )


                    # Check for NaN/Inf in loss before backward
                    if torch.isnan(combined_loss) or torch.isinf(combined_loss):
                       train_logger.error(f"Skipping Batch {batch_idx} due to NaN/Inf in loss.")
                       continue




    


                    #checks if it's invalid outputs in batch
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                       train_logger.error(f"Skipping Batch {batch_idx} due to NaN/Inf in outputs.")
                       continue


                    # Appending loss values for each Batch 
                    Append_loss_values_for_batch(mask_loss, hybrid_loss, combined_loss)



                    #Backward pass with the scaled loss.
                    scaler.scale(combined_loss).backward()

                    #Updates the model's weight with optimizer after accumulation steps.
                    if batch_idx % accumulation_steps == 0 or batch_idx == len(combined_train_loader):
                        # Unscale gradients for dynamic clipping
                        scaler.unscale_(optimizer)
                        
                        # Calculate the total gradient norm
                        total_norm = nn_utils.clip_grad_norm_(model.parameters(), float('inf'))
                        
                      
                       # Dynamic gradient clipping during accumulation steps
                        dynamic_clip_value = min(max(0.1 * total_norm, 0.1), gradient_clip_value)
                        dynamic_clip_value = max(dynamic_clip_value, 0.01)  # Enforce a minimum clip value
                        clipped_norm = nn_utils.clip_grad_norm_(model.parameters(), dynamic_clip_value)
                        train_logger.info(
                            f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx}/{len(combined_train_loader)}], "
                            f"Gradient Norm: {total_norm:.4f}, Clipped Norm: {clipped_norm:.4f}, "
                            f"Dynamic Clip Value: {dynamic_clip_value:.4f}"
                        )


                        # Apply gradient clipping using dynamic_clip_value
                        clipped_norm = nn_utils.clip_grad_norm_(model.parameters(), dynamic_clip_value)
                        
                        # Log the dynamic adjustment
                        train_logger.info(
                            f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx}/{len(combined_train_loader)}], "
                            f"Gradient Norm: {total_norm:.4f}, Clipped Norm: {clipped_norm:.4f}, "
                            f"Dynamic Clip Value: {dynamic_clip_value:.4f}"
                        )
                        
                        # Apply gradients
                        scaler.step(optimizer)
                        
                        # Update the scaler
                        scaler.update()



                    # Update after calculating losses
                    writer.add_scalar('Loss/Epoch_Mask', maskloss_avg, epoch)
                    writer.add_scalar('Loss/Epoch_Hybrid', hybridloss_avg, epoch)


                    running_loss += combined_loss.item() * accumulation_steps

        
                #LOGGING TO TENSORBOARD
                if representative_batch is not None:
                    mixture, predicted_vocals, target = representative_batch
                    log_spectrograms_to_tensorboard(
                        mixture[0].numpy(), sr=44100, tag=f"Epoch {epoch + 1} - Mixture Spectrogram", writer=writer, global_step=epoch
                    )
                    log_spectrograms_to_tensorboard(
                        predicted_vocals[0].numpy(), sr=44100, tag=f"Epoch {epoch + 1} - Estimated Vocal Spectrogram", writer=writer, global_step=epoch
                    )
                    log_spectrograms_to_tensorboard(
                        target[0].numpy(), sr=44100, tag=f"Epoch {epoch + 1} - Target Vocal Spectrogram", writer=writer, global_step=epoch
                    )

    
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Learning Rate', current_lr, epoch)

                train_logger.info(f"Current Learning Rate {current_lr}")
            
                
                if prev_epoch_loss is not None:  # Skip the first epoch since there's no previous loss to compare
                   loss_improvement = (prev_epoch_loss - avg_epoch_loss) / prev_epoch_loss * 100
                   train_logger.info( f"[Epoch Improvement] Epoch {epoch + 1}: Loss improved by {loss_improvement:.2f}% from previous epoch." )
                else:
                    train_logger.info(f"[Epoch Improvement] Epoch {epoch + 1}: No comparison (first epoch).")

        


                #Loss per epoch
                avg_epoch_loss = running_loss / len(combined_train_loader)
                train_logger.info(f"[Epoch Summary] Avg Training Loss: {avg_epoch_loss:.6f}")
                writer.add_scalar('Loss/Epoch_Combined', avg_epoch_loss, epoch)
                writer.add_scalar('Loss/Epoch_Mask', maskloss_avg, epoch)
                writer.add_scalar('Loss/Epoch_Hybrid', hybridloss_avg, epoch)


    



                #Fallback/best loss.. Checkpoint
                best_loss, trigger_times = save_model_checkpoint(avg_epoch_loss, epoch, model, best_loss, trigger_times)


                if trigger_times >= patience:
                   train_logger.info("Early stopping triggered.")
                   break


                if len(combined_val_loader) > 0:
                    avg_combined_loss, avg_mask_loss, avg_hybrid_loss = Validate_epoch(
                        model, combined_val_loader, criterion, device
                    )
                    if avg_combined_loss < best_val_loss:
                        best_val_loss = avg_combined_loss
                        val_checkpoint_path = os.path.join(Model_CheckPoint, f"best_val_model_epoch_{epoch + 1}.pth")
                        torch.save(model.state_dict(), val_checkpoint_path)
                        train_logger.info(f"Saved best validation model at epoch {epoch + 1} with loss {best_val_loss:.6f}")

                    train_logger.info(
                        f"[Validation] Epoch {epoch + 1}/{epochs}: "
                        f"Combined Loss={avg_combined_loss:.6f}, "
                        f"Mask Loss={avg_mask_loss:.6f}, "
                        f"Hybrid Loss={avg_hybrid_loss:.6f}"
                    )
                    scheduler.step(avg_combined_loss)
                    writer.add_scalar('loss/avg-Combinedloss [Evaluation]', avg_combined_loss, epoch)
                    writer.add_scalar('loss/Mask_loss [Evaluation]', avg_mask_loss, epoch)
                    writer.add_scalar('loss/HybridLosss [Evaluation]', avg_hybrid_loss, epoch)

                    # Save regular checkpoint
                if avg_combined_loss < best_val_loss:
                    best_val_loss = avg_combined_loss
                    best_model_path = os.path.join(Model_CheckPoint, f"best_model_epoch_{epoch + 1}.pth")
                    torch.save(model.state_dict(), best_model_path)
                    train_logger.info(f"Saved best validation model at epoch {epoch + 1} with loss {best_val_loss:.6f}")

                    




                #Logging memory after each 5 epochs.
                log_memory_after_index_epoch(epoch)
            


                #Get calculated avg loss from batches.
                maskloss_avg, hybridloss_avg, combined_loss_avg = Get_calculated_average_loss_from_batches()
            

                train_logger.info(
                f"[Epoch Summary] Epoch: {epoch + 1}/{epochs}, "
                f"Avg Combined Loss: {avg_epoch_loss:.6f}, MaskLoss: {maskloss_avg:.6f}, "
                f"Hybridloss: {hybridloss_avg:.6f}"
                )


                # Appending avg loss each epoch
                Append_loss_values_for_epoches(maskloss_avg, hybridloss_avg, combined_loss_avg, avg_epoch_loss)

                prev_epoch_loss = avg_epoch_loss
                
                print(f"Previous Epoch loss: {prev_epoch_loss}")
                
                train_logger.info(f"Previous epoch loss: {prev_epoch_loss}")

            #Creates Diagrams of loss during training for Batch & Epoch.
            create_loss_diagrams(loss_history_Batches,loss_history_Epoches)      


            #Training is completed. 
            training_completed()



            #Saves the best checkpoint model as the final model.
            if best_model_path is not None and  os.path.exists(best_model_path):
                os.makedirs(Final_model_path, exist_ok=True)
                final_pth = os.path.join(Final_model_path, "final_model_best_model.pth")
                shutil.copyfile(best_model_path,final_pth)
                train_logger.info(f"[Train] Copied best checkpoint and changed it's name to final_model_best_model.pth{best_model_path} -> {final_pth}")
            else: 
                save_final_model(model, Final_model_path)
                train_logger.info(f"[Train] Copied Last Checkpoint of the epoch loop. {best_model_path} -> {Final_model_path}")





        

        
        except Exception as e:
            train_logger.error(f"[Train] Error during training: {str(e)}")
    else:
        train_logger.info("Skipping training and starting fine_tuning.")

    #Fine_Tuning Config.
    fine_tuning_flag = False  

    if fine_tuning_flag:
        try:
            fine_tuned_model_path = os.path.join(fine_tuned_model_base_path, "fine_tuned_model.pth")
            pretrained_model_path = best_model_path if best_model_path else os.path.join(Final_model_path, "final_model.pth")
            fine_tune_model(
                pretrained_model_path=pretrained_model_path,
                fine_tuned_model_path=fine_tuned_model_path,
                Fine_tuned_training_loader=combined_train_loader,
                Finetuned_validation_loader=combined_val_loader,
                learning_rate=1e-3,
                fine_tune_epochs=6,
            )
            train_logger.info("Fine-tuning completed.")
        except Exception as e:
            train_logger.error(f"Error during fine-tuning: {e}")



if __name__ == "__main__":
   
    num_workers = 6
    batch_size = 4
    sampling_rate = 44100
    max_length_seconds = 10

    if 'combined_train_loader' not in globals():
         combined_train_loader, combined_val_loader = create_dataloaders(
            musdb18_dir=MUSDB18_dir,
            dsd100_dir=DSD100_dataset_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            sampling_rate=sampling_rate,
            max_length_seconds=max_length_seconds,
            max_files_train=None,
            max_files_val=None,
    )



   #Dataloader/Dataset

    train(load_model_path=r"C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_Weights\Fine_tuned\model.pth")
    
    


