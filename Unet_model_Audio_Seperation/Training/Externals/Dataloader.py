import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

# Adjust project root if necessary
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Datasets.Scripts.Dataset_Musdb18 import MUSDB18StemDataset
from Datasets.Scripts.Dataset_DSD100 import DSD100
from Training.Externals.Logger import setup_logger
from Training.Externals.utils import Return_root_dir


root_dir = Return_root_dir() #Gets the root directory
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DataLoader] Moving batch to {device}...")
    inputs_tensor = inputs_tensor.to(device)
    targets_tensor = targets_tensor.to(device)
    print(f"[DataLoader] Batch moved to {device}.")
    
    return inputs_tensor, targets_tensor

def create_dataloaders(
    musdb18_dir,
    dsd100_dir,
    batch_size=8,
    num_workers=0,  # Set num_workers to 0 to avoid multiprocessing issues
    sampling_rate=44100,
    max_length_seconds=15,
    max_files_train=None,
    max_files_val=None,
):
    musdb18_train_dataset = MUSDB18StemDataset(
        root_dir=musdb18_dir,
        subset='train',
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=max_files_train,
        validate_files=False,
    )

    musdb18_val_dataset = MUSDB18StemDataset(
        root_dir=musdb18_dir,
        subset='test',
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=max_files_val,
        validate_files=False,
    )

    dsd100_dev = DSD100(
        root_dir=dsd100_dir,
        subset='Dev',
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=max_files_train,
    )

    dsd100_test = DSD100(
        root_dir=dsd100_dir,
        subset='Test',
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=max_files_val,
    )

    combined_train_dataset = ConcatDataset([musdb18_train_dataset, dsd100_dev])
    combined_val_dataset = ConcatDataset([musdb18_val_dataset, dsd100_test])

    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=robust_collate_fn,
    )

    val_loader = DataLoader(
        combined_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=robust_collate_fn,
    )

    data_loader.info(f"Training dataset size: {len(combined_train_dataset)}")
    data_loader.info(f"Validation dataset size: {len(combined_val_dataset)}")
    print(f"Training dataset size: {len(combined_train_dataset)} samples")
    print(f"Validation dataset size: {len(combined_val_dataset)} samples")

    return train_loader, val_loader
