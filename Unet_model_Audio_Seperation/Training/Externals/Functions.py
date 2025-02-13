import torch
import gc
import os
import sys
import shutil
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Training.Externals.Logger import setup_logger
from Training.Externals.utils import Return_root_dir
root_dir = Return_root_dir() 
train_log_path = os.path.join(root_dir, "Model_performance_logg/log/Model_Training_logg.txt")
Function_logger = setup_logger('Functions.py',train_log_path)







def Early_break(trigger_times, patience):
    Function_logger.info(f"Trigger times: {trigger_times}, patience: {patience}")
    if trigger_times >= patience:
        Function_logger.info(f"Early stopping triggered.  The model is not getting any better!!!!")
        return True
    else: 
        return False



def training_completed():
    Function_logger.info("[Train] Training completed. Clearing memory cache now...")
    torch.cuda.empty_cache()
    gc.collect()
    print("Memory Cleared.")





def load_model_path_func(load_model_path, model_engine, model, device):
    print("Loading Model file....")

    if load_model_path and os.path.isdir(load_model_path):
        latest_file = os.path.join(load_model_path, "latest")

        if os.path.isfile(latest_file):
            try:
                with open(latest_file, "r") as f:
                    tag = f.read().strip()
                
                model_engine.load_checkpoint(load_model_path, tag=tag)
                Function_logger.info(f"Loaded model checkpoint from {load_model_path}")
                return  
            except Exception as e:
                Function_logger.warning(f"Failed to load DeepSpeed checkpoint, starting from scratch. Error: {e}")
        else:
            Function_logger.info(f"No 'latest' file found in {load_model_path}. Starting from scratch.")

    elif load_model_path and os.path.isfile(load_model_path):
        try:
            checkpoint = torch.load(load_model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            Function_logger.info("Loaded model from direct file")
            return  # Successfully loaded, return early.
        except Exception as e:
            Function_logger.warning(f"Failed to load PyTorch checkpoint, starting from scratch. Error: {e}")

    Function_logger.info("No checkpoint found. Starting training from scratch.")




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






###FINE TUNING###
def freeze_encoder(Fine_tune_logger,model):
    Fine_tune_logger.debug("Freezing encoder layers.")
    try:

        module = model.module if hasattr(model, 'module') else model
        encoder = getattr(module, 'encoder', None)
        if encoder is None:
            raise AttributeError("Model does not have an 'encoder' attribute.")
     
        if hasattr(encoder, '__iter__'):
            for layer in encoder:
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            for param in encoder.parameters():
                param.requires_grad = False
        encoder.eval()  
        Fine_tune_logger.info("Encoder layers frozen for fine-tuning.")
    except AttributeError as e:
        Fine_tune_logger.error(f"Error freezing encoder layers: {e}")
        raise




