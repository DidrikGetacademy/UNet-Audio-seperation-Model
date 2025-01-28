import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import torch.multiprocessing as mp

# Set the start method for multiprocessing to avoid conflicts
mp.set_start_method('spawn', force=True)
# Adjust project root if necessary
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Datasets.Scripts.Dataset_Musdb18 import MUSDB18StemDataset
from Datasets.Scripts.Dataset_DSD100 import DSD100
from Datasets.Scripts.Custom_audio_dataset import CustomAudioDataset
from Training.Externals.Logger import setup_logger
from Training.Externals.utils import Return_root_dir
root_dir = Return_root_dir() 
train_log_path = os.path.join(root_dir, "Model_performance_logg/log/Model_Training_logg.txt")
data_loader = setup_logger(  'dataloader', train_log_path)
TensorBoard_log_dir = os.path.join(root_dir, "Model_Performance_logg/Tensorboard")


def robust_collate_fn(batch):
    
    # 2D cropping and padding:
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.empty(0), torch.empty(0)

    inputs, targets = zip(*batch)

    # Find largest freq, time
    max_freq = max(x.size(-2) for x in inputs)
    max_time = max(x.size(-1) for x in inputs)

    padded_inputs = []
    padded_targets = []

    for inp, tgt in zip(inputs, targets):
        # Crop and pad inputs
        inp = F.pad(inp, (0, max_time - inp.size(-1), 0, max_freq - inp.size(-2)))
        tgt = F.pad(tgt, (0, max_time - tgt.size(-1), 0, max_freq - tgt.size(-2)))
        
        padded_inputs.append(inp)
        padded_targets.append(tgt)

    inputs_tensor = torch.stack(padded_inputs, dim=0)
    targets_tensor = torch.stack(padded_targets, dim=0)
    print(f"Inputs tensor min: {inputs_tensor.min()}, max: {inputs_tensor.max()}, mean: {inputs_tensor.mean()}")
    print(f"Targets tensor min: {targets_tensor.min()}, max: {targets_tensor.max()}, mean: {targets_tensor.mean()}")
    data_loader.debug(f"Inputs tensor min: {inputs_tensor.min()}, max: {inputs_tensor.max()}, mean: {inputs_tensor.mean()}")
    data_loader.debug(f"Targets tensor min: {targets_tensor.min()}, max: {targets_tensor.max()}, mean: {targets_tensor.mean()}")

    if inputs_tensor.is_cpu:
        inputs_tensor = inputs_tensor.pin_memory()
        targets_tensor = targets_tensor.pin_memory()
   
    print(f"Batch sizes: inputs - {len(inputs)}, targets - {len(targets)}")
    print(f"Padded inputs shape: {inputs_tensor.shape}, Padded targets shape: {targets_tensor.shape}")
    data_loader.debug(f"Padded inputs shape: {inputs_tensor.shape}, Padded targets shape: {targets_tensor.shape}")
    data_loader.debug(f"Batch sizes: inputs - {len(inputs)}, targets - {len(targets)}")

    return inputs_tensor, targets_tensor

def create_dataloaders(
    custom_dataset_dir,
    musdb18_dir,
    dsd100_dir,
    batch_size=0,
    num_workers=4,
    sampling_rate=44100,
    max_length_seconds=10,
    max_files_train=None,
    max_files_val=None,
    val_ratio=0.2,
):
    # Splitting max_files dynamically for training and validation
    if max_files_train:
        num_val_files_musdb18 = int(max_files_train * val_ratio)
        num_train_files_musdb18 = max_files_train - num_val_files_musdb18
    else:
        num_val_files_musdb18 = None
        num_train_files_musdb18 = None

    if max_files_val:
        num_val_files_dsd100 = int(max_files_val * val_ratio)
        num_train_files_dsd100 = max_files_val - num_val_files_dsd100
    else:
        num_val_files_dsd100 = None
        num_train_files_dsd100 = None

    # --- TRAINING ---
    musdb18_train_dataset = MUSDB18StemDataset(
        root_dir=musdb18_dir,
        subset="train",
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=num_train_files_musdb18,
        validate_files=False,
    )

    dsd100_train_dataset = DSD100(
        root_dir=dsd100_dir,
        subset="Dev",
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=num_train_files_dsd100,
    )


    # --- VALIDATION ---
    musdb18_val_dataset = MUSDB18StemDataset(
        root_dir=musdb18_dir,
        subset="train",
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=num_val_files_musdb18,
        validate_files=False,
    )

    dsd100_val_dataset = DSD100(
        root_dir=dsd100_dir,
        subset="Dev",
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=num_val_files_dsd100,
    )



    # --- EVALUATION ---
    musdb18_eval_dataset = MUSDB18StemDataset(
        root_dir=musdb18_dir,
        subset="test",
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=max_files_val,
        validate_files=False,
    )

    dsd100_eval_dataset = DSD100(
        root_dir=dsd100_dir,
        subset="Test",
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=max_files_val,
    )

    # Create DataLoaders
    train_loader = DataLoader( #(combined training datasets).
        ConcatDataset([musdb18_train_dataset, dsd100_train_dataset]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=robust_collate_fn,
    )

    val_loader = DataLoader( #val_loader (combined validation datasets) after each epoch.
        ConcatDataset([musdb18_val_dataset, dsd100_val_dataset]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=robust_collate_fn,
    )

    eval_loader = DataLoader( #(combined evaluation datasets) to assess the model's final performance on unseen data.
        ConcatDataset([musdb18_eval_dataset, dsd100_eval_dataset]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=robust_collate_fn,
    )

    return train_loader, val_loader, eval_loader
