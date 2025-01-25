import torch
import gc
import os
import sys
import matplotlib.pyplot as plt
import shutil
import platform
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Training.Externals.Logger import setup_logger
from Training.Externals.Memory_debugging import (  clear_memory_before_training )
from Training.Fine_Tuned_model import fine_tune_model 
from Training.Externals.utils import Return_root_dir


root_dir = Return_root_dir() #Gets the root directory
Model_CheckPoint = os.path.join(root_dir, "Model_Weights/CheckPoints")
train_log_path = os.path.join(root_dir, "Model_performance_logg/log/Model_Training_logg.txt")
train_logger = setup_logger('train',train_log_path)
os.makedirs(Model_CheckPoint, exist_ok=True)





def save_representative_batch(representative_batch, save_dir="/mnt/c/Users/didri/Desktop/UNet-Models/Unet_model_Audio_Seperation/Model_Performance_logg/Diagrams", file_prefix="batch"):
    inputs, predicted_vocals, targets = representative_batch

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    for i, (data, title) in enumerate(zip(
        [inputs[0], predicted_vocals[0], targets[0]], 
        ["Input", "Predicted_Vocals", "Target"]
    )):
        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.specgram(data.numpy(), NFFT=2048, Fs=44100, noverlap=1024)
        plt.colorbar()
        file_path = os.path.join(save_dir, f"{file_prefix}_{title.replace(' ', '_')}.png")
        plt.savefig(file_path)
        plt.close()  # Close the plot to free memory
        print(f"Saved plot: {file_path}")



def training_completed():
    train_logger.info("[Train] Training completed. Clearing memory cache now...")
    torch.cuda.empty_cache()
    gc.collect()

def load_model_path_func(load_model_path, model_engine, model, device):

    if load_model_path:
        if os.path.isdir(load_model_path):
            try:
                # Check for 'latest' file to determine the checkpoint tag
                latest_file = os.path.join(load_model_path, "latest")
                if not os.path.isfile(latest_file):
                    raise FileNotFoundError(f"'latest' file not found in {load_model_path}. Ensure it exists.")

                with open(latest_file, "r") as f:
                    tag = f.read().strip()
                
                print(f"[Train] Attempting to load DeepSpeed checkpoint from {load_model_path} with tag '{tag}'...")
                train_logger.info(f"[Train] Attempting to load DeepSpeed checkpoint from {load_model_path} with tag '{tag}'...")

                # Load checkpoint using DeepSpeed
                model_engine.load_checkpoint(load_model_path, tag=tag)
                train_logger.info(f"[Train] Successfully loaded DeepSpeed checkpoint from {load_model_path} with tag '{tag}'.")
                print(f"[Train] Successfully loaded DeepSpeed checkpoint from {load_model_path} with tag '{tag}'.")
            except Exception as e:
                train_logger.error(f"[Train] Failed to load DeepSpeed checkpoint from {load_model_path}: {e}")
                print(f"[Train] Failed to load DeepSpeed checkpoint from {load_model_path}: {e}")
                raise RuntimeError(f"Failed to load DeepSpeed checkpoint: {e}")
        elif os.path.isfile(load_model_path):
            try:
                # Load standard PyTorch checkpoint
                print(f"[Train] Attempting to load PyTorch checkpoint from {load_model_path}...")
                train_logger.info(f"[Train] Attempting to load PyTorch checkpoint from {load_model_path}...")

                checkpoint = torch.load(load_model_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                train_logger.info(f"[Train] Successfully loaded PyTorch checkpoint from {load_model_path}.")
                print(f"[Train] Successfully loaded PyTorch checkpoint from {load_model_path}.")
            except Exception as e:
                train_logger.error(f"[Train] Failed to load PyTorch checkpoint from {load_model_path}: {e}")
                print(f"[Train] Failed to load PyTorch checkpoint from {load_model_path}: {e}")
                raise RuntimeError(f"Failed to load PyTorch checkpoint: {e}")
        else:
            train_logger.warning(f"[Train] Provided model path {load_model_path} does not exist. Starting training from scratch.")
            print(f"[Train] Provided model path {load_model_path} does not exist. Starting training from scratch.")
    else:
        train_logger.info("[Train] No model path provided. Starting training from scratch.")
        print("[Train] No model path provided. Starting training from scratch.")

    # Clear memory after loading
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
    device = inputs.device  # Log the device where the inputs are located
    if inputs is None or targets is None:
       train_logger.warning(f"[Train] Skipping batch {batch_idx} due to None data.")
       train_logger.debug(f"no valid data in {batch_idx}")
    else: 
        train_logger.debug(f"batch: {batch_idx} is valid on device {device}")
        print(f"batch: {batch_idx} is valid on device {device}")



def print_inputs_targets_shape(inputs, targets, batch_idx):
    if batch_idx <= 2:
       train_logger.debug(f"Batch {batch_idx}: Inputs shape={inputs.shape}, Targets shape={targets.shape}") 
       train_logger.debug(f"Inputs min={inputs.min().item():.4f}, max={inputs.max().item():.4f}")
       train_logger.debug(f"Targets min={targets.min().item():.4f}, max={targets.max().item():.4f}")
       print(f"Inputs and Targets moved to device: {inputs.device}")

        


def Early_break(trigger_times, patience, train_logger):
    if trigger_times >= patience:
        train_logger.info("Early stopping triggered.")
        return True
    else: 
        return False


def Automatic_Fine_Tune(combined_val_loader, combined_train_loader, fine_tuned_model_base_path, Final_model_path, best_model_path):
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


def save_best_model(model_engine, best_model_path, final_model_dir, train_logger):
    try:
        if model_engine and best_model_path is None:
            # Save model state using model_engine
            os.makedirs(final_model_dir, exist_ok=True)
            final_model_path = os.path.join(final_model_dir, "final_model_best_model.pth")
            torch.save(model_engine.state_dict(), final_model_path)
            train_logger.info(f"[Train] Model saved using model_engine at: {final_model_path}")
            print(f"[Train] Model saved using model_engine: {final_model_path}")
        elif best_model_path is not None and os.path.exists(best_model_path):
            # Save the best checkpoint file
            os.makedirs(final_model_dir, exist_ok=True)
            final_model_path = os.path.join(final_model_dir, "final_model_best_model.pth")
            shutil.copyfile(best_model_path, final_model_path)
            train_logger.info(f"[Train] Copied best checkpoint to final model: {best_model_path} -> {final_model_path}")
            print(f"[Train] Best model saved as: {final_model_path}")
        else:
            train_logger.warning(f"[Train] Neither model_engine nor best_model_path provided.")
            print(f"[Train] No valid best model found to save.")
    except Exception as e:
        train_logger.error(f"[Train] Error saving best model: {str(e)}")
        print(f"[Train] Error saving best model: {str(e)}")



def Return_root_dir():
    if platform.system()  == "Windows":
        root_dir = r"C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation"
    elif platform.system() == "Linux":
        root_dir = "/mnt/c/Users/didri/Desktop/UNet-Models/Unet_model_Audio_Seperation"
    else: 
        raise OSError("Unsupported Platform")
    return root_dir
        