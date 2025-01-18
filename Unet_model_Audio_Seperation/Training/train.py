# train.py

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils as nn_utils
import os
import sys
import gc
import shutil
from torch import autocast, GradScaler
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Training.Externals.Dataloader import create_dataloaders
from Training.Externals.Memory_debugging import log_memory_usage
from Training.Fine_Tuned_model import fine_tune_model 
from Training.Externals.Logger import setup_logger
from Training.Loss_Diagram_Values import (
    plot_loss_curves_Training_script_epoches,
    plot_loss_curves_Training_script_Batches,
)

from Model_Architecture.model import UNet






train_logger = setup_logger( 'train', r'C:\Users\didri\Desktop\UNet Models\UNet_vocal_isolation_model\Model_performance_logg\log\Model_Training_logg.txt')
Final_model_path = r"C:\Users\didri\Desktop\UNet Models\UNet_vocal_isolation_model\Model_weights\CheckPoints"
diagramdirectory = r"C:\Users\didri\Desktop\UNet Models\UNet_vocal_isolation_model\Model_performance_logg\Diagrams"
fine_tuned_model_base_path = r"C:\Users\didri\Desktop\UNet Models\UNet_vocal_isolation_model\Model_weights\Fine_tuned"
Model_CheckPoint = r"C:\Users\didri\Desktop\UNet Models\UNet_vocal_isolation_model\Model_weights\CheckPoints"

# Create necessary directories
os.makedirs(diagramdirectory, exist_ok=True)
os.makedirs(fine_tuned_model_base_path, exist_ok=True)
os.makedirs(Model_CheckPoint, exist_ok=True)
os.makedirs(Final_model_path, exist_ok=True)



# Initialize global variables for loss tracking
loss_history_Epoches = {
    "l1": [],            # L1 loss per epoch
    "spectral": [],      # Spectral loss (STFT MSE) per epoch
    "combined": [],      # Combined loss (weighted sum of L1 and spectral) per epoch
    "Total_loss_per_epoch": [],  # Total loss per epoch
}

loss_history_Batches = {
    "l1": [],            # L1 loss per batch
    "spectral/mse": [],  # Spectral loss or MSE per batch
    "combined": [],      # Combined loss per batch
}



def create_loss_diagrams(loss_history_Batches, loss_history_Epoches):
    train_logger.info("Creating Diagrams now")
    batches_figpath = os.path.join(diagramdirectory, "loss_curves_training_batches.png")
    epoch_figpath = os.path.join(diagramdirectory, "loss_curves_training_epoches.png")
    plot_loss_curves_Training_script_Batches(loss_history_Batches, out_path=batches_figpath)
    plot_loss_curves_Training_script_epoches(loss_history_Epoches, out_path=epoch_figpath)





def clear_memory_before_training():
    log_memory_usage(tag="Clearing memory before training")
    torch.cuda.empty_cache()
    gc.collect()
    log_memory_usage(tag="Memory after clearing it...")





def training_completed():
    train_logger.info("[Train] Training completed. Clearing memory cache now...")
    torch.cuda.empty_cache()
    gc.collect()




def log_memory_after_index_epoch(epoch):
    if epoch == 0 or (epoch + 1) % 5 == 0:  
                log_memory_usage(tag=f"After Epoch {epoch + 1}")




def save_final_model(model, Final_model_path):
    os.makedirs(os.path.dirname(Final_model_path), exist_ok=True) 
    torch.save(model.state_dict(), os.path.join(Final_model_path, "final_model.pth"))
    train_logger.info(f"[Train] Final model saved at {Final_model_path}")




def load_model_path_func(load_model_path, model, device):
    if load_model_path is not None:
        if os.path.exists(load_model_path):
            model.load_state_dict(torch.load(load_model_path, map_location=device, weights_only=True))
            train_logger.info(f"Loaded model from {load_model_path}")
        else:
            train_logger.info(f"[Train] Model path {load_model_path} does not exist. Starting from scratch.")
        clear_memory_before_training()
    else:
        train_logger.info("[Train] No existing model path provided. Starting from scratch.")
        clear_memory_before_training()




def save_model_checkpoint(avg_epoch_loss, epoch, model, best_loss,  trigger_times):
    global best_model_path 
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        best_model_path = os.path.join(Model_CheckPoint, f"best_model_epoch-{epoch}.pth")
        torch.save(model.state_dict(), best_model_path)
        train_logger.info(f"[Train] New best model saved at {best_model_path} with loss {best_loss:.6f}")
        trigger_times = 0  
    else:
        train_logger.info(f"[Train] No improvement in loss for epoch {epoch + 1} with loss: {avg_epoch_loss}. Best loss remains {best_loss:.6f}. Trigger_times: {trigger_times}")
        trigger_times += 1

    return best_loss, trigger_times






def Model_Structure_Information(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_logger.info(f"Total number of parameters: {total_params}")
    train_logger.info(f"Trainable parameters: {trainable_params}") 
    train_logger.info(f"Model architecture:\n{model}")





def Append_loss_values_for_batch(l1_val, stft_val, combined_loss):
    loss_history_Batches["l1"].append(l1_val.item())
    loss_history_Batches["spectral/mse"].append(stft_val.item())
    loss_history_Batches["combined"].append(combined_loss.item())




def Append_loss_values_for_epoches(l1_val_avg,stft_val_avg,combined_loss_avg,avg_epoch_loss):
    loss_history_Epoches["l1"].append(l1_val_avg)
    loss_history_Epoches["spectral"].append(stft_val_avg)
    loss_history_Epoches["combined"].append(combined_loss_avg)
    loss_history_Epoches["Total_loss_per_epoch"].append(avg_epoch_loss)





def dataset_sample_information(musdb18_Train_Dataloader, musdb18_Evaluation_Dataloader):
    try:
        train_logger.info(f"Training dataset: {len(musdb18_Train_Dataloader)} batches")
        train_logger.info(f"Validation dataset: {len(musdb18_Evaluation_Dataloader)} batches")
        for batch_idx, (samples_mixture, vocal_mixture) in enumerate(musdb18_Train_Dataloader):
            train_logger.debug(f"Batch {batch_idx} -> Mixture shape: {samples_mixture.shape}, Vocal shape: {vocal_mixture.shape}")
            if batch_idx == 0:
                break
        data_iter = iter(musdb18_Train_Dataloader)
        samples_mixture, vocal_mixture = next(data_iter)
        train_logger.debug(f"Sample Mixture shape: {samples_mixture.shape}, Sample Vocal Mixture shape: {vocal_mixture.shape}")
        print(f"Sample Mixture shape: {samples_mixture.shape}, Sample Vocal Mixture shape: {vocal_mixture.shape}")
    except StopIteration:
        train_logger.error("[Dataset Sample Info] DataLoader is empty. Cannot fetch samples.")
    except Exception as e:
        train_logger.error(f"[Dataset Sample Info] Error fetching samples: {str(e)}")





def Get_calculated_average_loss_from_batches():
    l1_val_avg = sum(loss_history_Batches["l1"]) / len(loss_history_Batches["l1"]) if loss_history_Batches["l1"] else 0
    stft_val_avg = sum(loss_history_Batches["spectral/mse"]) / len(loss_history_Batches["spectral/mse"]) if loss_history_Batches["spectral/mse"] else 0
    combined_loss_avg = sum(loss_history_Batches["combined"]) / len(loss_history_Batches["combined"]) if loss_history_Batches["combined"] else 0
    return l1_val_avg, stft_val_avg, combined_loss_avg





def check_inputs_targets_dataset(inputs, targets, batch_idx):
    print(f" batch: {batch_idx} inputs: {inputs.shape} - targets: {targets.shape}")
    if inputs is None or targets is None:
       train_logger.warning(f"[Train] Skipping batch {batch_idx} due to None data.")
       train_logger.debug(f"no valid data in {batch_idx}")
    else: 
        train_logger.debug(f"batch: {batch_idx} is valid.")




def print_inputs_targets_shape(inputs, targets, batch_idx):
    if batch_idx <= 2:
       train_logger.debug(f"Batch {batch_idx}: Inputs shape={inputs.shape}, Targets shape={targets.shape}") 
       train_logger.debug(f"Inputs min={inputs.min().item():.4f}, max={inputs.max().item():.4f}")
       train_logger.debug(f"Targets min={targets.min().item():.4f}, max={targets.max().item():.4f}")




def Validate_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_batch_idx, (inputs,targets) in enumerate(val_loader):
           inputs, targets = inputs.to(device), targets.to(device)
           outputs = model(inputs)
           combined_loss, l1_loss, stft_loss = criterion(outputs,targets)
           val_loss += combined_loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss



class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        l1 = self.l1_loss(pred, target)
        n_fft = min(2048, pred.size(-1))
        hop_length = 512
        window = torch.hann_window(n_fft, device=pred.device)
        batch_size = pred.size(0)
        stft_loss = 0.0
        for i in range(batch_size):
            pred_tensor = pred[i].squeeze()
            target_tensor = target[i].squeeze()
            min_len = min(pred_tensor.size(-1), target_tensor.size(-1))
            pred_tensor = pred_tensor[..., :min_len]
            target_tensor = target_tensor[..., :min_len]
            pred_stft = torch.stft(pred_tensor, n_fft=n_fft, hop_length=hop_length, return_complex=True, window=window)
            target_stft = torch.stft(target_tensor, n_fft=n_fft, hop_length=hop_length, return_complex=True, window=window)
            pred_stft_log = torch.log1p(torch.abs(pred_stft))
            target_stft_log = torch.log1p(torch.abs(target_stft))
            stft_loss_batch = self.mse_loss(pred_stft_log, target_stft_log)
            stft_loss += stft_loss_batch
            train_logger.debug(f"[HybridLoss] Batch {i}: STFT MSE={stft_loss_batch.item():.6f}")
        stft_loss /= batch_size
        combined_loss = 0.3 * l1 + 0.7 * stft_loss
        train_logger.debug(f"[HybridLoss] L1={l1.item():.6f}, STFT={stft_loss.item():.6f}, Combined={combined_loss.item():.6f}")
        return combined_loss, l1, stft_loss


def train(load_model_path=r'C:\Users\didri\Desktop\UNet Models\UNet_vocal_isolation_model\Model_weights\Pre_trained\best_model_epoch-19.pth',start_training=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_logger.info(f"[Train] Using device: {device}")

   #Training config
    batch_size = 6
    desired_effective_batch_size = 128
    accumulation_steps = desired_effective_batch_size // batch_size
    learning_rate = 1e-7
    gradient_clip_value = 1  
    epochs = 40
    num_workers = 0
    patience = 7
    best_loss = float('inf')
    trigger_times = 0
    prev_epoch_loss = None
    best_model_path = None 
    sampling_rate = 44100
    max_length_seconds = 10
 

   #Dataloader/Dataset
    MUSDB18_dir = r'C:\Users\didri\Desktop\UNet Models\UNet_vocal_isolation_model\Datasets\Dataset_Audio_Folders\musdb18'
    DSD100_dataset_dir =r'C:\Users\didri\Desktop\UNet Models\UNet_vocal_isolation_model\Datasets\Dataset_Audio_Folders\DSD100'

    combined_train_loader, combined_val_loader = create_dataloaders(
        musdb18_dir=MUSDB18_dir,
        dsd100_dir=DSD100_dataset_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        sampling_rate=sampling_rate,
        max_length_seconds=max_length_seconds,
        max_files_train=None,
        max_files_val=None
    )



    #UNet - Model initialization, Optimizer Config, Custom Hybridloss function, Gradscaler.
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = HybridLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate, 
        weight_decay=1e-4, 
        betas=(0.9, 0.999),
        eps=1e-7
        )

  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
          optimizer, 
          mode='min',  # Reduce LR when a monitored metric (validation loss) stops decreasing
          factor=0.5,  # Reduce LR by this factor
          patience=3,  # Number of epochs to wait before reducing LR
          threshold=1e-5,  # Minimum change to qualify as an improvement
          verbose=True
          )


    scaler = GradScaler()

    Effective_batch_size = batch_size * accumulation_steps


    if len(combined_val_loader) > 0:
        print(f"Combined_val_loader: {len(combined_val_loader)}")
    else: 
        print("Combined_val_loader is 0")
    if len(combined_train_loader) > 0:
        print(f"combined_train_loader: {len(combined_train_loader)}")
    else: 
        print("Combined_train_loader is 0")


    #Prininting Model Structure/Information
    Model_Structure_Information(model)



    # Loading model if exists or creating a new one. 
    load_model_path_func(load_model_path, model, device)
     


    #Prints sample of the dataset.
    dataset_sample_information(combined_train_loader,combined_val_loader)


    for inputs, targets in combined_train_loader:
       print(f"Inputs shape: {inputs.shape}") 
       print(f"Targets shape: {targets.shape}")  
       break

    if start_training:
        try:
            train_logger.info(f"Training started with configuration: Batch size={batch_size}, Effective batch size: {Effective_batch_size} Learning rate={learning_rate}, Epochs={epochs}")
            
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                train_logger.info(f"[Train] Epoch {epoch + 1}/{epochs} started.")

    
                for batch_idx, (inputs, targets) in enumerate(combined_train_loader, start=1):
       
                    #Resets gradients. 
                    optimizer.zero_grad()
                



                    #Data check & logging.
                    check_inputs_targets_dataset(inputs, targets, batch_idx)
                    print_inputs_targets_shape(inputs, targets, batch_idx)



                    #Moves the data to device.
                    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                    with autocast(device_type='cuda', enabled=(device.type == 'cuda')):

                        print(f"Batch {batch_idx} -> Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")

                        outputs = model(inputs)


                        print(f"Batch {batch_idx} -> Outputs shape: {outputs.shape}")
                        
                        combined_loss, l1_val, stft_val = criterion(outputs, targets)

                    #Normalizing the loss.
                    combined_loss = combined_loss / accumulation_steps


                    # Check for NaN/Inf in loss before backward
                    if torch.isnan(combined_loss) or torch.isinf(combined_loss):
                       train_logger.error(f"Skipping Batch {batch_idx} due to NaN/Inf in loss.")
                       continue


                    #Backward pass with the scaled loss.
                    scaler.scale(combined_loss).backward()



                    train_logger.info(
                    f"[Batch] Epoch: {epoch + 1}/{epochs}, Batch: {batch_idx}/{len(combined_train_loader)}, "
                    f"Combined Loss: {combined_loss.item():.6f} (Summen av L1 og STFT), "
                    f"L1 Loss: {l1_val.item():.6f} (Amplitudetap), "
                    f"STFT Loss: {stft_val.item():.6f} (Spektraltap)"
                    )
    


                    #checks if it's invalid outputs in batch
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                       train_logger.error(f"Skipping Batch {batch_idx} due to NaN/Inf in outputs.")
                       continue


                    # Appending loss values for each Batch 
                    Append_loss_values_for_batch(l1_val, stft_val, combined_loss)




                    #Updates the model's weight with optimizer after accumulation steps.
                    if batch_idx % accumulation_steps == 0 or batch_idx == len(combined_train_loader):
                       if gradient_clip_value is not None:
                          #Unscaling gradient before clipping.
                          scaler.unscale_(optimizer) 
                          total_norm = nn_utils.clip_grad_norm_(model.parameters(),gradient_clip_value)
                          train_logger.info(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx}/{len(combined_train_loader)}], Gradient Norm: {total_norm:.4f}")
                       else: 
                            total_norm = nn_utils.clip_grad_norm_(model.parameters(), float('inf'))  # Compute norm without clipping
                            train_logger.info(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx}/{len(combined_train_loader)}], Gradient Norm: {total_norm:.4f}")


                    #Applying gradients.
                    scaler.step(optimizer) 

                    #Updates the scaler.
                    scaler.update() 
            
                    


                    running_loss += combined_loss.item()
        

    
                current_lr = optimizer.param_groups[0]['lr']
                train_logger.info(f"Current Learning Rate {current_lr}")
            
                
                if prev_epoch_loss is not None:  # Skip the first epoch since there's no previous loss to compare
                   loss_improvement = (prev_epoch_loss - avg_epoch_loss) / prev_epoch_loss * 100
                   train_logger.info( f"[Epoch Improvement] Epoch {epoch + 1}: Loss improved by {loss_improvement:.2f}% from previous epoch." )
                else:
                    train_logger.info(f"[Epoch Improvement] Epoch {epoch + 1}: No comparison (first epoch).")

        


                #Loss per epoch
                avg_epoch_loss = running_loss / len(combined_train_loader)
                train_logger.info(f"[Epoch Summary] Avg Training Loss: {avg_epoch_loss:.6f}")



              



                #Fallback/best loss.. Checkpoint
                best_loss, trigger_times = save_model_checkpoint(avg_epoch_loss, epoch, model, best_loss, trigger_times)
                if trigger_times >= patience:
                   train_logger.info("Early stopping triggered.")
                   break


                if len(combined_val_loader) > 0:
                    avg_val_loss = Validate_epoch(model,combined_val_loader,criterion,device)
                    train_logger.info(
                    f"[Validation] Epoch {epoch + 1}/{epochs}, "
                    f"[Validation] Avg Validation Loss:  {avg_val_loss:.6f}")
                    #Step the scheduler based on validation loss
                    scheduler.step(avg_val_loss)
                else:
                    train_logger.info("[Validation] Skipping validation: No validation dataset provided.")
                    




                #Logging memory after each 5 epochs.
                log_memory_after_index_epoch(epoch)
            


                #Get calculated avg loss from batches.
                l1_val_avg, stft_val_avg, combined_loss_avg = Get_calculated_average_loss_from_batches()
            

                train_logger.info(
                f"[Epoch Summary] Epoch: {epoch + 1}/{epochs}, "
                f"Avg Combined Loss: {avg_epoch_loss:.6f}, Avg L1 Loss: {l1_val_avg:.6f}, "
                f"Avg STFT Loss: {stft_val_avg:.6f}"
                )


                # Appending avg loss each epoch
                Append_loss_values_for_epoches(l1_val_avg, stft_val_avg, combined_loss_avg, avg_epoch_loss)

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
    try:
        fine_tuning_flag = True  
        if fine_tuning_flag:
            fine_tuned_model_path = os.path.join(fine_tuned_model_base_path, "fine_tuned_model.pth")
            pretrained_model_path = os.path.join(Final_model_path,"final_model.pth")
            fine_tune_model(
                pretrained_model_path=pretrained_model_path,
                fine_tuned_model_path=fine_tuned_model_path,
            )
            train_logger.info("Fine_tuning Completed.")
    except Exception as e:
            train_logger.error(f"Error during fine-tuning: {str(e)}")


if __name__ == "__main__":
    train(load_model_path=r"C:\Users\didri\Desktop\UNet Models\UNet_vocal_isolation_model\Model_weights\Pre_trained\best_model_epoch-19.pth")
