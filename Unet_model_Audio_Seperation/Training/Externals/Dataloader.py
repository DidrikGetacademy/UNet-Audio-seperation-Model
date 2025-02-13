import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Datasets.Scripts.Dataset_Musdb18 import MUSDB18StemDataset
from Datasets.Scripts.Dataset_DSD100 import DSD100
from Datasets.Scripts.Custom_audio_dataset import CustomAudioDataset
from Training.Externals.Logger import setup_logger
from Training.Externals.utils import Return_root_dir
from Datasets.Scripts.Dataset_utils import Convert_spectrogram_to_audio

root_dir = Return_root_dir() 
train_log_path = os.path.join(root_dir, "Model_Performance_logg/log/Dataloader.txt")
data_loader = setup_logger('dataloader', train_log_path)
BatchCount = 0

def robust_collate_fn(batch):
    check_batch = float('inf')
    batch = [item for item in batch if item is not None]
    
    if not batch:
        raise ValueError("Empty batch")
    
    check_batch = 0
    for item in batch:
        if check_batch < 1:
            data_loader.info(f"Batch item: {item}")
            check_batch += 1

    inputs, targets = zip(*batch)
    
    max_freq = max(x.size(-2) for x in inputs)
    max_time = max(x.size(-1) for x in inputs)

    padded_inputs = []
    padded_targets = []

    for inp, tgt in zip(inputs, targets):
        inp = F.pad(inp, (0, max_time - inp.size(-1), 0, max_freq - inp.size(-2)))
        tgt = F.pad(tgt, (0, max_time - tgt.size(-1), 0, max_freq - tgt.size(-2)))
        padded_inputs.append(inp)
        padded_targets.append(tgt)

    inputs_tensor = torch.stack(padded_inputs, dim=0)
    targets_tensor = torch.stack(padded_targets, dim=0)
    
    # Logging tensor statistics for debugging
    data_loader.info(f"[Dataloader-robust_collate_fn] Inputs tensor min: {inputs_tensor.min()}, max: {inputs_tensor.max()}, mean: {inputs_tensor.mean()}")
    data_loader.info(f"[Dataloader-robust_collate_fn] Targets tensor min: {targets_tensor.min()}, max: {targets_tensor.max()}, mean: {targets_tensor.mean()}")
    data_loader.info(f"[Dataloader-robust_collate_fn] Padded inputs shape: {inputs_tensor.shape}, Padded targets shape: {targets_tensor.shape}")
    data_loader.info(f"[Dataloader-robust_collate_fn] Batch sizes: inputs - {len(inputs)}, targets - {len(targets)}\n")

    if inputs_tensor.device.type == "cpu":
        data_loader.info("Tensors are on CPU, skipping pin_memory.")
    
    if BatchCount <= 2:
        print(f"Converting spectrogram to audio now.... BATCH COUNT:{BatchCount}")
        Convert_spectrogram_to_audio(audio_path="/mnt/c/Users/didri/Desktop/Programmering/ArtificalintelligenceModels/UNet-Model_Vocal_Isolation/Unet_model_Audio_Seperation/audio_logs/Dataloading", predicted_vocals=None,targets=targets_tensor[0],inputs=inputs_tensor[0],outputs=None,)
        

    
    return inputs_tensor, targets_tensor


MUSDB18_dir = os.path.join(root_dir, "Datasets/Dataset_Audio_Folders/Training_MUSDB18_dataset")
DSD100_dataset_dir = os.path.join(root_dir, "Datasets/Dataset_Audio_Folders/Training_DSD100_dataset")
custom_dataset_dir = os.path.join(root_dir, "Datasets/Dataset_Audio_Folders/Evaluation_Test_Dataset")


### DATALOADER 1 ###
def create_dataloader_training(
    musdb18_dir=MUSDB18_dir,
    dsd100_dir=DSD100_dataset_dir,
    batch_size=0,
    num_workers=6,
    sampling_rate=44100,
    max_length_seconds=10,
    max_files_train=None,
    max_files_val=None,
    val_ratio=0.2,
):
    if max_files_train:
        num_val_files_musdb18 = int(max_files_train * val_ratio)
        num_train_files_musdb18 = max_files_train - num_val_files_musdb18
        data_loader.info(f"[splitting files/max_files_train] Number of files in MUSDB18 for training: {num_train_files_musdb18}\n"
                         f"[splitting files/max_files_train] Number of files in MUSDB18 for validation: {num_val_files_musdb18}\n")
    else:
        num_val_files_musdb18 = len(os.listdir(musdb18_dir))  # Using total files if None
        num_train_files_musdb18 = num_val_files_musdb18  # All files for training if None
        data_loader.info(f"[None/max_files_train] Number of files in MUSDB18 for training: {num_train_files_musdb18}\n"
                         f"[None/max_files_train] Number of files in MUSDB18 for validation: {num_val_files_musdb18}\n")

    if max_files_val:
        num_val_files_dsd100 = int(max_files_val * val_ratio)
        num_train_files_dsd100 = max_files_val - num_val_files_dsd100
        data_loader.info(f"[splitting files/max_files_val] Number of files in DSD100 for training: {num_train_files_dsd100}\n"
                         f"[splitting files/max_files_val] Number of files in DSD100 for validation: {num_val_files_dsd100}\n")
    else:
        num_val_files_dsd100 = len(os.listdir(dsd100_dir))  # Using total files if None
        num_train_files_dsd100 = num_val_files_dsd100  # All files for training if None
        data_loader.info(f"[None/max_files_val] Number of files in DSD100 for training: {num_train_files_dsd100}\n"
                         f"[None/max_files_val] Number of files in DSD100 for validation: {num_val_files_dsd100}\n")

    # --- TRAINING DATASETS ---
    musdb18_train_dataset = MUSDB18StemDataset(
        root_dir=musdb18_dir,
        subset="train",
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=num_train_files_musdb18,
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


    # --- VALIDATION DATASETS ---
    musdb18_val_dataset = MUSDB18StemDataset(
        root_dir=musdb18_dir,
        subset="train",
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=num_val_files_musdb18,
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



    #DATALOADERS
    Training_loader = DataLoader(
        ConcatDataset([musdb18_train_dataset, dsd100_train_dataset]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=robust_collate_fn,
        pin_memory=True 
    )

    Validation_loader = DataLoader(
        ConcatDataset([dsd100_val_dataset, musdb18_val_dataset]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=robust_collate_fn,
        pin_memory=True  
    )

    return Training_loader, Validation_loader









####FINE_TUNING#####
def create_dataloader_Fine_tuning(
    musdb18_dir=MUSDB18_dir,
    dsd100_dir=DSD100_dataset_dir,
    batch_size=0,
    num_workers=4,
    sampling_rate=44100,
    max_files_finetuning_train=None,
    max_files_fine_tuning_validation=None,
    val_ratio=0.2,
    max_length_seconds = 10
):
    if max_files_finetuning_train:
        num_val_files_musdb18 = int(max_files_finetuning_train * val_ratio)
        num_train_files_musdb18 = max_files_finetuning_train - num_val_files_musdb18
        data_loader.info(f"[splitting files/max_files_train] Number of files in MUSDB18 for training: {num_train_files_musdb18}\n"
                         f"[splitting files/max_files_train] Number of files in MUSDB18 for validation: {num_val_files_musdb18}\n")
    else:
        num_val_files_musdb18 = len(os.listdir(musdb18_dir))  # Using total files if None
        num_train_files_musdb18 = num_val_files_musdb18  # All files for training if None
        data_loader.info(f"[None/max_files_train] Number of files in MUSDB18 for training: {num_train_files_musdb18}\n"
                         f"[None/max_files_train] Number of files in MUSDB18 for validation: {num_val_files_musdb18}\n")

    if max_files_fine_tuning_validation:
        num_val_files_dsd100 = int(max_files_fine_tuning_validation * val_ratio)
        num_train_files_dsd100 = max_files_fine_tuning_validation - num_val_files_dsd100
        data_loader.info(f"[splitting files/max_files_val] Number of files in DSD100 for training: {num_train_files_dsd100}\n"
                         f"[splitting files/max_files_val] Number of files in DSD100 for validation: {num_val_files_dsd100}\n")
    else:
        num_val_files_dsd100 = len(os.listdir(dsd100_dir))  # Using total files if None
        num_train_files_dsd100 = num_val_files_dsd100  # All files for training if None
        data_loader.info(f"[None/max_files_val] Number of files in DSD100 for training: {num_train_files_dsd100}\n"
                         f"[None/max_files_val] Number of files in DSD100 for validation: {num_val_files_dsd100}\n")

    # --- TRAINING DATASETS ---
    musdb18_train_dataset = MUSDB18StemDataset(
        root_dir=musdb18_dir,
        subset="train",
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=num_train_files_musdb18,
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


    # --- VALIDATION DATASETS ---
    musdb18_val_dataset = MUSDB18StemDataset(
        root_dir=musdb18_dir,
        subset="train",
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=num_val_files_musdb18,
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

    #DATALOADERS
    Fine_tuning_training_loader = DataLoader(
        ConcatDataset([musdb18_train_dataset, dsd100_train_dataset]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=robust_collate_fn,
        pin_memory=True 
    )

    Fine_tuning_validation_loader = DataLoader(
        ConcatDataset([dsd100_val_dataset, musdb18_val_dataset]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=robust_collate_fn,
        pin_memory=True  
    )

    return Fine_tuning_training_loader, Fine_tuning_validation_loader















######## EVALUATION #####
def create_dataloader_EVALUATION(
    musdb18_dir=MUSDB18_dir,
    dsd100_dir=DSD100_dataset_dir,
    customDataset_dir=custom_dataset_dir,
    batch_size=0,
    num_workers=0,
    max_length_seconds=10,
    sampling_rate=44100,
    max_files_val=None,
):
    # --- EVALUATION DATASETS ---
    musdb18_eval_dataset = MUSDB18StemDataset(
        root_dir=musdb18_dir,
        subset="test",
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=max_files_val,
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

    custom_eval_dataset = CustomAudioDataset(
        root_dir=customDataset_dir,
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=max_files_val,
    )


    #DATALOADER
    Evaluation_Loader = DataLoader(
        ConcatDataset([musdb18_eval_dataset, dsd100_eval_dataset, custom_eval_dataset]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=robust_collate_fn,
        pin_memory=True 
    )

    return Evaluation_Loader