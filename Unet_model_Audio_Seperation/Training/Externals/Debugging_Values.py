import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Training.Externals.Logger import setup_logger
from Training.Externals.utils import Return_root_dir

root_dir = Return_root_dir() #Gets the root directory
Debug_value = setup_logger('debugging_values',os.path.join(root_dir,"Model_Performance_logg/log/Debugging_values.txt"))



#CHECK IF INPUTS OR TARGETS ARE VALID OR NONE
def check_inputs_targets_dataset(inputs, targets, batch_idx):
    device = inputs.device 
    if inputs is None or targets is None:
       Debug_value.debug(f"[Train] Skipping batch {batch_idx} due to None data. inputs: {inputs.shape}, targets: {targets.shape}")
    else: 
        Debug_value.debug(f"batch: {batch_idx} is valid on device {device}")


#Debugs the inputs and targets shape. 
def print_inputs_targets_shape(inputs, targets, batch_idx):
    if batch_idx <= 2:
       Debug_value.debug(f"Batch {batch_idx}: Inputs shape={inputs.shape}, Targets shape={targets.shape}") 
       Debug_value.debug(f"Inputs min={inputs.min().item():.4f}, max={inputs.max().item():.4f}, Targets min={targets.min().item():.4f}, max={targets.max().item():.4f}")



#Checks the Samples in the dataset before training.
def dataset_sample_information(musdb18_Train_Dataloader, musdb18_Evaluation_Dataloader):
    try:
        Debug_value.info(f"Training dataset: {len(musdb18_Train_Dataloader)} batches")
        Debug_value.info(f"Validation dataset: {len(musdb18_Evaluation_Dataloader)} batches")
        check_nr = 0
        for batch_idx, (samples_mixture, vocal_mixture) in enumerate(musdb18_Train_Dataloader):
            if check_nr > 5:
                break
            Debug_value.debug(f"Batch {batch_idx} -> Mixture shape: {samples_mixture.shape}, Vocal shape: {vocal_mixture.shape}")
            if batch_idx == 0:
                break
        check_nr += 1
        data_iter = iter(musdb18_Train_Dataloader)
        samples_mixture, vocal_mixture = next(data_iter)
        Debug_value.debug(f"Sample Mixture shape: {samples_mixture.shape}, Sample Vocal Mixture shape: {vocal_mixture.shape}")
        print(f"Sample Mixture shape: {samples_mixture.shape}, Sample Vocal Mixture shape: {vocal_mixture.shape}")
    except StopIteration:
        Debug_value.error("[Dataset Sample Info] DataLoader is empty. Cannot fetch samples.")
    except Exception as e:
        Debug_value.error(f"[Dataset Sample Info] Error fetching samples: {str(e)}")

