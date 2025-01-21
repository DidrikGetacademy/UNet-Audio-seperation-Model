import torch
import gc
import os
import sys
from Training.Externals.Memory_debugging import log_memory_usage
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Training.Externals.Logger import setup_logger
train_logger = setup_logger( 'train', r'C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_performance_logg\log\Model_Training_logg.txt')
from Training.Externals.Memory_debugging import (  clear_memory_before_training )


Model_CheckPoint = r"C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_Weights\CheckPoints"
os.makedirs(Model_CheckPoint, exist_ok=True)

    
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




def save_final_model(model, Final_model_path):
    os.makedirs(os.path.dirname(Final_model_path), exist_ok=True) 
    torch.save(model.state_dict(), os.path.join(Final_model_path, "final_model.pth"))
    train_logger.info(f"[Train] Final model saved at {Final_model_path}")



#VALIDATION FUNCTION
def Validate_epoch(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    val_combined_loss = 0
    val_mask_loss = 0
    val_hybrid_loss = 0

    with torch.no_grad():
        for val_batch_idx, (inputs, targets) in enumerate(val_loader):
      
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
          
            predicted_mask, outputs = model(inputs.to(device, non_blocking=True))

            if predicted_mask.size() != targets.size():
                raise ValueError(f"Validation shape mismatch: predicted_mask={predicted_mask.size()}, targets={targets.size()}")

           
            combined_loss, mask_loss, hybrid_loss = criterion(
                predicted_mask.to(device), 
                inputs.to(device), 
                targets.to(device)
            )

    
            val_combined_loss += combined_loss.item()
            val_mask_loss += mask_loss.item()
            val_hybrid_loss += hybrid_loss.item()


    avg_combined_loss = val_combined_loss / len(val_loader)
    avg_mask_loss = val_mask_loss / len(val_loader)
    avg_hybrid_loss = val_hybrid_loss / len(val_loader)

    return avg_combined_loss, avg_mask_loss, avg_hybrid_loss




def training_completed():
    train_logger.info("[Train] Training completed. Clearing memory cache now...")
    torch.cuda.empty_cache()
    gc.collect()



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



def Model_Structure_Information(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_logger.info(f"Total number of parameters: {total_params}")
    train_logger.info(f"Trainable parameters: {trainable_params}") 
    train_logger.info(f"Model architecture:\n{model}")


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



#CHECK IF INPUTS OR TARGETS ARE VALID OR NONE
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


        