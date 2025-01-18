import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Training.Externals.Dataloader import create_dataloaders
from Training.train import train
from Training.Fine_Tuned_model import fine_tune_model

def start():
    MUSDB18_dir = r'C:\Users\didri\Desktop\UNet Models\UNet_vocal_isolation_model\Datasets\Dataset_Audio_Folders\musdb18'
    
    fine_tuned_model_base_path = r"C:\Users\didri\Desktop\UNet Models\UNet_vocal_isolation_model\Model_weights\Fine_tuned"
    Final_model_path = r"C:\Users\didri\Desktop\UNet Models\UNet_vocal_isolation_model\Model_weights\CheckPoints"
    fine_tuned_model_path = os.path.join(fine_tuned_model_base_path, "fine_tuned_model.pth")
    pretrained_model_path = os.path.join(Final_model_path,"final_model.pth")
    DSD100_dataset_dir =r'C:\Users\didri\Desktop\UNet Models\UNet_vocal_isolation_model\Datasets\Dataset_Audio_Folders\DSD100'
    batch_size = 8
    sampling_rate = 44100
    max_length_seconds = 10
    num_workers = 6
    epochs= 30


    # Initialize datasets once
    combined_train_loader, combined_val_loader = create_dataloaders(
        musdb18_dir=MUSDB18_dir,
        dsd100_dir=DSD100_dataset_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        sampling_rate=sampling_rate,
        max_length_seconds=max_length_seconds
    )

# Pass these DataLoaders to both training and fine-tuning
#train(combined_train_loader, combined_val_loader, epochs, batch_size)
    fine_tune_model(pretrained_model_path, fine_tuned_model_path, combined_train_loader, combined_val_loader)
if __name__ == '__main__':
                  start()