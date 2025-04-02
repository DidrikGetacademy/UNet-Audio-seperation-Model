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
MUSDB18_dir = os.path.join(root_dir, "Datasets/Dataset_Audio_Folders/Training_MUSDB18_dataset")
DSD100_dataset_dir = os.path.join(root_dir, "Datasets/Dataset_Audio_Folders/Training_DSD100_dataset")
custom_dataset_dir = os.path.join(root_dir, "Datasets/Dataset_Audio_Folders/Evaluation_Test_Dataset")
data_loader_path = os.path.join(root_dir, "Model_Performance_logg/log/Dataloader.txt")
data_loader = setup_logger('dataloader', data_loader_path)
conversion_batch_count  = 0
check_batch = 0


def robust_collate_fn(batch):
    global conversion_batch_count
    global check_batch

    num_none = sum(1 for item in batch if item is None)
    data_loader.info(f"Batch contains {num_none} None items out of {len(batch)}")

    Valid_batch = [item for item in batch if item is not None]
    total_items = len(batch)
    invalid_items = total_items - len(Valid_batch) 
     

    data_loader.info(
        f"Valid items in batch: {len(Valid_batch)} out of {total_items}\n"
        f"Invalid items in batch: {invalid_items}"
        )
    

    if not Valid_batch:
        data_loader.warning("Skipping empty batch.")
        return torch.empty(0, 1, 513, 948), torch.empty(0, 1, 513, 948)  
    

    inputs, targets = zip(*Valid_batch)
    
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
    data_loader.info(f"\n[Dataloader-robust_collate_fn] Inputs tensor min: {inputs_tensor.min()}, max: {inputs_tensor.max()}, mean: {inputs_tensor.mean()}\n")
    data_loader.info(f"[Dataloader-robust_collate_fn] Targets tensor min: {targets_tensor.min()}, max: {targets_tensor.max()}, mean: {targets_tensor.mean()}\n")
    data_loader.info(f"[Dataloader-robust_collate_fn] Padded inputs shape: {inputs_tensor.shape}, Padded targets shape: {targets_tensor.shape}\n")
    data_loader.info(f"[Dataloader-robust_collate_fn] Batch sizes: inputs - {len(inputs)}, targets - {len(targets)}\n")

    if inputs_tensor.device.type == "cpu":
        data_loader.info("Tensors are on CPU, skipping pin_memory.")
    
    if conversion_batch_count  <= 2:
        print(f"Converting spectrogram to audio now.... BATCH COUNT:{conversion_batch_count}\n")
        Convert_spectrogram_to_audio(audio_path="/mnt/c/Users/didri/Desktop/Programmering/ArtificalintelligenceModels/UNet-Model_Vocal_Isolation/Unet_model_Audio_Seperation/audio_logs/Dataloading", predicted_vocals=None,targets=targets_tensor[0],inputs=inputs_tensor[0],outputs=None,)
        conversion_batch_count  += 1

    return inputs_tensor, targets_tensor




### DATALOADER 1 ###
def create_dataloader_training(
    musdb18_dir=MUSDB18_dir,
    dsd100_dir=DSD100_dataset_dir,
    batch_size=0,
    num_workers=6,
    sampling_rate=44100,
    max_length_seconds=10,
    max_files_train=None,
    max_files_val=30,
):
  

    # --- TRAINING DATASETS ---
    musdb18_train_dataset = MUSDB18StemDataset(
        root_dir=musdb18_dir,
        subset="train",
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=max_files_train,
    )

    dsd100_train_dataset = DSD100(
        root_dir=dsd100_dir,
        subset="Dev",
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=max_files_train,
    )


    # --- VALIDATION DATASETS ---
    musdb18_val_dataset = MUSDB18StemDataset(
        root_dir=musdb18_dir,
        subset="test",
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=max_files_val,
    )

    dsd100_val_dataset = DSD100(
        root_dir=dsd100_dir,
        subset="test",
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=max_files_val,
    )


    #DATALOADERS
    Training_loader = DataLoader(
        ConcatDataset([musdb18_train_dataset]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=robust_collate_fn,
        pin_memory=True 
    )

    Validation_loader = DataLoader(
        ConcatDataset([musdb18_val_dataset]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=robust_collate_fn,
        pin_memory=True  
    )

    return Training_loader, Validation_loader





######## EVALUATION #####
def create_dataloader_EVALUATION(
    musdb18_dir=MUSDB18_dir,
    dsd100_dir=DSD100_dataset_dir,
    customDataset_dir=custom_dataset_dir,
    batch_size=0,
    num_workers=6,
    max_length_seconds=11,
    sampling_rate=44100,
    max_files_val=30,
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
        ConcatDataset([dsd100_eval_dataset]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=robust_collate_fn,
        pin_memory=True 
    )

    return Evaluation_Loader







from torch.utils.data import DataLoader, ConcatDataset

def create_dataloader_Fine_tuning(
    musdb18_dir=MUSDB18_dir,
    dsd100_dir=DSD100_dataset_dir,
    batch_size=0,
    num_workers=6,
    sampling_rate=44100,
    max_files_train=50,
    max_files_finetuning_val=20,
    max_length_seconds=11
):

    # --- TRAINING DATASETS ---
    
    musdb18_train_dataset = MUSDB18StemDataset(
        root_dir=musdb18_dir,
        subset="train",
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=max_files_train,
    )

    dsd100_train_dataset = DSD100(
        root_dir=dsd100_dir,
        subset="Dev",
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=max_files_train,
    )

    # --- VALIDATION DATASETS ---
    musdb18_val_dataset = MUSDB18StemDataset(
        root_dir=musdb18_dir,
        subset="test",
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=max_files_finetuning_val, 
    )

    dsd100_val_dataset = DSD100(
        root_dir=dsd100_dir,
        subset="test",
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        max_length_seconds=max_length_seconds,
        max_files=max_files_finetuning_val,  
    )

    # --- DATA LOADERS ---
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
        ConcatDataset([musdb18_val_dataset, dsd100_val_dataset]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=robust_collate_fn,
        pin_memory=True  
    )

    return Fine_tuning_training_loader, Fine_tuning_validation_loader
