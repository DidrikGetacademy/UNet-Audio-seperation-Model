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
from Fine_tuning.Fine_Tuned_model import fine_tune_model 
from Training.Externals.utils import Return_root_dir


root_dir = Return_root_dir() 
train_log_path = os.path.join(root_dir, "Model_performance_logg/log/Model_Training_logg.txt")
Function_logger = setup_logger('Functions.py',train_log_path)







def load_model_path_func(load_model_path, model_engine, model, device):
    print("Loading Model file....")
    if load_model_path:
        if os.path.isdir(load_model_path):
            try:
                latest_file = os.path.join(load_model_path, "latest")
                if not os.path.isfile(latest_file):
                    raise FileNotFoundError(f"'latest' file not found in {load_model_path}. Ensure it exists.")

                with open(latest_file, "r") as f:
                    tag = f.read().strip()
                
                model_engine.load_checkpoint(load_model_path, tag=tag)
                Function_logger.info(f"loaded model engine from {load_model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load DeepSpeed checkpoint: {e}")
        elif os.path.isfile(load_model_path):
            try:
                checkpoint = torch.load(load_model_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                Function_logger.info("loaded model engine direct file")
            except Exception as e:
                raise RuntimeError(f"Failed to load PyTorch checkpoint: {e}")
        else:
            Function_logger.debug(f"[Train] Provided model path {load_model_path} does not exist. Starting training from scratch.")
    else:
        Function_logger.info("[Train] No model path provided. Starting training from scratch.")











#Stops training if trigger hit the amount of times configured. 
def Early_break(trigger_times, patience):
    if trigger_times >= patience:
        Function_logger.info(f"Early stopping triggered.")
        return True
    else: 
        return False



def training_completed():
    Function_logger.info("[Train] Training completed. Clearing memory cache now...")
    torch.cuda.empty_cache()
    gc.collect()
    print("Memory Cleared.")



#Saving the best model out of the best checkpoint, else it just saves normal.
def save_best_model(model_engine, best_model_path, final_model_dir):
    try:
        if model_engine and best_model_path is None:
            os.makedirs(final_model_dir, exist_ok=True)
            final_model_path = os.path.join(final_model_dir, "final_model_best_model.pth")
            torch.save(model_engine.state_dict(), final_model_path)
            Function_logger.info(f"[Train] Model saved using model_engine at: {final_model_path}")
            print(f"[Train] Model saved using model_engine: {final_model_path}")
        elif best_model_path is not None and os.path.exists(best_model_path):
            os.makedirs(final_model_dir, exist_ok=True)
            final_model_path = os.path.join(final_model_dir, "final_model_best_model.pth")
            shutil.copyfile(best_model_path, final_model_path)
            Function_logger.info(f"[Train] Copied best checkpoint to final model: {best_model_path} -> {final_model_path}")

        else:
            Function_logger.debug(f"[Train] Neither model_engine nor best_model_path provided.")
    
    except Exception as e:
        Function_logger.error(f"[Train] Error saving best model: {str(e)}")


