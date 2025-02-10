import os
import sys
import torch 
import torchaudio
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Training.Externals.Logger import setup_logger
from Training.Externals.utils import Return_root_dir
from Datasets.Scripts.Dataset_utils import spectrogram_to_waveform
root_dir = Return_root_dir() #Gets the root directory
Debug_value = setup_logger('debugging_values',os.path.join(root_dir,"Model_Performance_logg/log/Debugging_values.txt"))



def save_audio_files_from_model_dataset_mask(predicted_vocals,targets,inputs,outputs):
        predicted_waveform = spectrogram_to_waveform(predicted_vocals.to("cpu"))
        target_waveform = spectrogram_to_waveform(targets.to("cpu"))
        input_waveform= spectrogram_to_waveform(inputs.to("cpu"))
        output_waveform = spectrogram_to_waveform(outputs.to("cpu"))
        torchaudio.save("/mnt/c/Users/didri/Desktop/Programmering/ArtificalintelligenceModels/UNet-Model_Vocal_Isolation/Unet_model_Audio_Seperation/audio_logs/predictions/predicted__audio.wav", predicted_waveform.unsqueeze(0), sample_rate=44100)
        torchaudio.save("/mnt/c/Users/didri/Desktop/Programmering/ArtificalintelligenceModels/UNet-Model_Vocal_Isolation/Unet_model_Audio_Seperation/audio_logs/outputs/output_audio.wav", output_waveform.unsqueeze(0), sample_rate=44100)
        torchaudio.save("/mnt/c/Users/didri/Desktop/Programmering/ArtificalintelligenceModels/UNet-Model_Vocal_Isolation/Unet_model_Audio_Seperation/audio_logs/inputs/input_audio.wav", input_waveform.unsqueeze(0), sample_rate=44100)
        torchaudio.save("/mnt/c/Users/didri/Desktop/Programmering/ArtificalintelligenceModels/UNet-Model_Vocal_Isolation/Unet_model_Audio_Seperation/audio_logs/targets/target_audio.wav", target_waveform.unsqueeze(0), sample_rate=44100)


def dataset_sample_information(musdb18_Train_Dataloader, musdb18_Evaluation_Dataloader):
    try:
        print("Function: [dataset_sample_information]")
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



def log_first_2_batches_inputs_targets(batch_idx,train_logger,inputs,targets):
        if batch_idx < 2:
            train_logger.info(
            f"#####INPUTS & TARGETS VALUE [2 BATCHES]####\n"
            f"Batch {batch_idx}: Mixture shape={inputs.shape}, Target shape={targets.shape}"
           )


def log_first_2_batches_outputs_inputs_targets_predicted_mask(batch_idx,outputs,inputs,targets,predicted_mask,train_logger,predicted_vocals):
         if batch_idx < 2:
            print("Function: [log_first_2_batches_outputs_inputs_targets_predicted_mask]")
            train_logger.info(
            f"####OUTPUTS, INPUTS, TARGETS,PREDICTEDMASK [2 BATCHES]####\n"
            f"Batch {batch_idx}: Mask range: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}"
            f"Batch {batch_idx}: Inputs shape={inputs.shape}, Targets shape={targets.shape}, Predicted Mask shape={predicted_mask.shape}, Outputs shape={outputs.shape}"
            f"Mask min={predicted_mask.min().item()}, max={predicted_mask.max().item()}"
            f"[After Mask Application] Predicted vocals min: {predicted_vocals.min()}, max: {predicted_vocals.max()}\n\n\n"
            )





def print_inputs_targets_shape(inputs, targets, batch_idx):
    if batch_idx <= 2:
       print("Function: [print_inputs_targets_shape]")
       Debug_value.debug(f"Batch {batch_idx}: Inputs shape={inputs.shape}, Targets shape={targets.shape}") 
       Debug_value.debug(f"Inputs min={inputs.min().item():.4f}, max={inputs.max().item():.4f}, Targets min={targets.min().item():.4f}, max={targets.max().item():.4f}")





def check_Nan_Inf_loss(combined_loss,batch_idx,outputs):
     if torch.isnan(combined_loss) or torch.isinf(combined_loss):
                       Debug_value.error(f"Skipping Batch {batch_idx} due to NaN/Inf in loss.\n")
                      

     if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                       Debug_value.error(f"Skipping Batch {batch_idx} due to NaN/Inf in outputs.\n")
                       


def logging_avg_loss_epoch(epoch, prev_epoch_loss, epochs, avg_epoch_loss, maskloss_avg, hybridloss_avg, train_logger):
    train_logger.info(
    f"####LOGGING AVERAGE LOSS EPOCH###\n"
    f"[Epoch Summary] Epoch: {epoch + 1}/{epochs}, "
    f"Avg Combined Loss: {avg_epoch_loss:.6f}, MaskLoss: {maskloss_avg:.6f}, "
    f"Hybridloss: {hybridloss_avg:.6f}"
    f"Previous Epoch loss: {prev_epoch_loss}"
    )


#Ikke i bruk 
def logging_avg_loss_batches(train_logger,epoch,epochs,avg_combined_loss,avg_mask_loss,avg_hybrid_loss):
    train_logger.info(
    f"###LOGGING AVERAGE LOSS BATCHES###\n"
    f"[Validation] Epoch {epoch + 1}/{epochs}: "
    f"Combined Loss={avg_combined_loss:.6f}, "
    f"Mask Loss={avg_mask_loss:.6f}, "
    f"Hybrid Loss={avg_hybrid_loss:.6f}"
    )



def prev_epoch_loss_log(train_logger,prev_epoch_loss,avg_epoch_loss,epoch):
        if prev_epoch_loss is None:
            prev_epoch_loss = avg_epoch_loss  
        if prev_epoch_loss is not None:  
            loss_improvement = (prev_epoch_loss - avg_epoch_loss) / prev_epoch_loss * 100
            train_logger.info( 
                              f"[Epoch Improvement] Epoch {epoch + 1}: Loss improved during training by {loss_improvement:.2f}% from previous epoch, " )
        else:

            train_logger.info(f"[Epoch Improvement] Epoch {epoch + 1}: No comparison (first epoch).")

 


