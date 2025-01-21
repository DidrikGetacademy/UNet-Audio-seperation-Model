import torch
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from torch.utils.data import DataLoader, ConcatDataset
from Datasets.Scripts.Dataset_Musdb18 import MUSDB18StemDataset
from Datasets.Scripts.Dataset_DSD100 import DSD100
from Training.Externals.Logger import setup_logger



data_loader = setup_logger('dataloader',  r'C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_performance_logg\log\Model_Training_logg.txt')



def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None

    inputs, targets = zip(*batch)
    
    max_length = max(inp.size(-1) for inp in inputs)

    padded_inputs = [torch.nn.functional.pad(inp, (0, max_length - inp.size(-1))) for inp in inputs]
    padded_targets = [torch.nn.functional.pad(tgt, (0, max_length - tgt.size(-1))) for tgt in targets]


    inputs_tensor = torch.stack(padded_inputs)
    targets_tensor = torch.stack(padded_targets)

    return inputs_tensor, targets_tensor


def create_dataloaders(
    musdb18_dir,
    dsd100_dir,
    batch_size=4,
    num_workers=6,
    sampling_rate=44100,
    max_length_seconds=10,
    max_files_train=None,
    max_files_val=None
):


    musdb18_train_dataset = MUSDB18StemDataset(
        root_dir=musdb18_dir,
        subset='train',
        sr=sampling_rate,
        n_fft=2048,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=max_files_train,
        validate_files=False,
    )

    musdb18_val_dataset = MUSDB18StemDataset(
        root_dir=musdb18_dir,
        subset='test',
        sr=sampling_rate,
        n_fft=2048,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=max_files_val,
        validate_files=False,
    )

    dsd100_dev = DSD100(
        root_dir=dsd100_dir,
        subset='Dev',
        sr=sampling_rate,
        n_fft=2048,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=max_files_train,
    )

    dsd100_test = DSD100(
        root_dir=dsd100_dir,
        subset='Test',
        sr=sampling_rate,
        n_fft=2048,
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
        collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        combined_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )

    data_loader.info(f"Training dataset size: {len(combined_train_dataset)}")
    data_loader.info(f"Validation dataset size: {len(combined_val_dataset)}")
    print(f"Training dataset size: {len(combined_train_dataset)} samples")
    print(f"Validation dataset size: {len(combined_val_dataset)} samples")
    return train_loader, val_loader
