# dataloader.py
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

data_loader = setup_logger(
    'dataloader',
    r'C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_performance_logg\log\Model_Training_logg.txt'
)

def robust_collate_fn(batch):
    #2D cropping and padding:
      #- Drops None items
      #- Finds max freq, max time across the batch
      #- Crops bigger shapes down, pads smaller shapes up
      #- Stacks final shape => [B, 1, freq, time]

    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None

    inputs, targets = zip(*batch)

    # find largest freq, time
    max_freq = max(x.size(-2) for x in inputs)
    max_time = max(x.size(-1) for x in inputs)

    padded_inputs = []
    padded_targets = []

    for inp, tgt in zip(inputs, targets):
        freq_in, time_in = inp.size(-2), inp.size(-1)
        # Crop freq if > max
        if freq_in > max_freq:
            inp = inp[..., :max_freq, :]
        # Crop time if > max
        if time_in > max_time:
            inp = inp[..., :max_time]

        freq_in, time_in = inp.size(-2), inp.size(-1)
        freq_pad = max_freq - freq_in
        time_pad = max_time - time_in
        inp_padded = F.pad(inp, (0, time_pad, 0, freq_pad))

        # same for target
        freq_tgt, time_tgt = tgt.size(-2), tgt.size(-1)
        if freq_tgt > max_freq:
            tgt = tgt[..., :max_freq, :]
        if time_tgt > max_time:
            tgt = tgt[..., :max_time]

        freq_tgt, time_tgt = tgt.size(-2), tgt.size(-1)
        freq_pad_t = max_freq - freq_tgt
        time_pad_t = max_time - time_tgt
        tgt_padded = F.pad(tgt, (0, time_pad_t, 0, freq_pad_t))

        padded_inputs.append(inp_padded)
        padded_targets.append(tgt_padded)

    inputs_tensor = torch.stack(padded_inputs, dim=0)
    targets_tensor = torch.stack(padded_targets, dim=0)

    return inputs_tensor, targets_tensor

def create_dataloaders(
    musdb18_dir,
    dsd100_dir,
    batch_size=2,
    num_workers=0,
    sampling_rate=44100,
    max_length_seconds=5,
    max_files_train=None,
    max_files_val=None,
):
    
    #Creates train/val DataLoaders from MUSDB18 + DSD100 Will produce spectrograms of shape [1, freq, time].
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
        #prefetch_factor=2 
    )

    val_loader = DataLoader(
        combined_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=robust_collate_fn, 
        #prefetch_factor=2 
    )

    data_loader.info(f"Training dataset size: {len(combined_train_dataset)}")
    data_loader.info(f"Validation dataset size: {len(combined_val_dataset)}")
    print(f"Training dataset size: {len(combined_train_dataset)} samples")
    print(f"Validation dataset size: {len(combined_val_dataset)} samples")

    return train_loader, val_loader
