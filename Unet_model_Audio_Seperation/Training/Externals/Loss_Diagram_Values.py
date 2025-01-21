import matplotlib.pyplot as plt
import sys
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Training.Externals.Logger import setup_logger
train_logger = setup_logger('train', r'C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_performance_logg\log\Model_Training_logg.txt')
diagramdirectory = r"C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_performance_logg\Diagrams"
os.makedirs(diagramdirectory, exist_ok=True)







##TRAINING-EPOCHES###### TRAINING-EPOCHES ####
def plot_loss_curves_Training_script_epoches(loss_history_Epoches, out_path="loss_curves_training_epoches.png"):
    epochs_count = len(loss_history_Epoches["Total_loss_per_epoch"])

    print(f"Epochs Count: {epochs_count}, Total Loss Entries: {len(loss_history_Epoches['Total_loss_per_epoch'])}")
    train_logger.info(f"Epochs Count: {epochs_count}, Total Loss Entries: {len(loss_history_Epoches['Total_loss_per_epoch'])}")

    if len(loss_history_Epoches["Total_loss_per_epoch"]) != epochs_count:
        print(f"Error: Mismatch between epochs ({epochs_count}) and Total_loss_per_epoch ({len(loss_history_Epoches['Total_loss_per_epoch'])}).")
        train_logger.error(f"Epoch Loss Mismatch: Expected {epochs_count}, Found {len(loss_history_Epoches['Total_loss_per_epoch'])}.") 
        return  
    
    epochs = list(range(1, epochs_count + 1))

    plt.figure(figsize=(12, 6))

    # Dynamisk skalering for y-aksen
    y_min = min(
        min(loss_history_Epoches["mask_loss"], default=0),
        min(loss_history_Epoches["hybrid_loss"], default=0),
        min(loss_history_Epoches["combined"], default=0),
        min(loss_history_Epoches["Total_loss_per_epoch"], default=0),
    )
    y_max = max(
        max(loss_history_Epoches["mask_loss"], default=1),
        max(loss_history_Epoches["hybrid_loss"], default=1),
        max(loss_history_Epoches["combined"], default=1),
        max(loss_history_Epoches["Total_loss_per_epoch"], default=1),
    )
    plt.ylim([y_min * 0.9, y_max * 1.1])  # Juster y-aksen for bedre synlighet

    if len(loss_history_Epoches["mask_loss"]) > 0:
        plt.plot(epochs, loss_history_Epoches["mask_loss"], label="mask_loss", color="blue", linewidth=2)

    if len(loss_history_Epoches["hybrid_loss"]) > 0:
        plt.plot(epochs, loss_history_Epoches["hybrid_loss"], label="hybrid_loss", color="green", linewidth=2)

    if len(loss_history_Epoches["combined"]) > 0:
        plt.plot(epochs, loss_history_Epoches["combined"], label="Combined-loss", color="yellow", linewidth=2)

    plt.plot(epochs, loss_history_Epoches["Total_loss_per_epoch"], label="Total_loss_per_epoch", color="purple", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Time (Epoch)")
    plt.legend()
    plt.grid(True)

    plt.xticks(range(1, epochs_count + 1))
    plt.xlim([1, epochs_count])

    # Status-tekst
    if epochs_count > 0:
        Final_loss = loss_history_Epoches["combined"][-1]
        if Final_loss < 0.01:
            status_text = f"Epoch {epochs_count}: Outstanding! Loss is exceptionally low ({Final_loss:.4f}). Great job!"
        elif Final_loss < 0.05:
            status_text = f"Epoch {epochs_count}: Excellent progress! Loss is impressively low ({Final_loss:.4f}). Keep it up!"
        elif Final_loss < 0.1:
            status_text = f"Epoch {epochs_count}: Good performance. Loss is low ({Final_loss:.4f}). Stay consistent!"
        elif Final_loss < 0.2:
            status_text = f"Epoch {epochs_count}: Decent. Loss is moderate ({Final_loss:.4f}). Fine-tuning might help!"
        elif Final_loss < 0.3:
            status_text = f"Epoch {epochs_count}: Loss is slightly high ({Final_loss:.4f}). Consider optimizing your model or data."
        else:
            status_text = f"Epoch {epochs_count}: Loss is high ({Final_loss:.4f}). Investigate potential issues with training or model."
    else:
        status_text = "No data available for evaluation."

    plt.text(0.5, 0.5, status_text, fontsize=12, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.7))

    # Lagre plottet
    base, ext = os.path.splitext(out_path)
    if ext.lower() not in [".png", ".jpg", ".jpeg", ".svg", ".pdf"]:
        out_path = base + ".png"

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print("Generated plot_loss_curves_Training_script_epoches diagram.")







###TRAINING-BATCHES####
def plot_loss_curves_Training_script_Batches(loss_history_Batches, out_path="loss_curves_training_batches.png"):
    batch_count = len(loss_history_Batches["combined"])
    batch_range = range(1, batch_count + 1)

    print(f"Batch Count: {batch_count}, Combined Loss Entries: {len(loss_history_Batches['combined'])}")

    plt.figure(figsize=(14, 7))  # Larger figure for better readability

    # Plot L1-loss
    if len(loss_history_Batches["mask_loss"]) > 0:
        plt.plot(batch_range, loss_history_Batches["mask_loss"], label="mask_loss", color="blue", linewidth=1.5, alpha=0.8)

    # Plot Spectral/MSE-loss
    if len(loss_history_Batches["hybrid_loss"]) > 0:
        plt.plot(batch_range, loss_history_Batches["hybrid_loss"], label="hybrid_loss", color="green", linewidth=1.5, alpha=0.8)

    # Plot Combined-loss
    plt.plot(batch_range, loss_history_Batches["combined"], label="Combined-loss", color="purple", linewidth=2, alpha=0.9)

    # Adding labels and title
    plt.xlabel("Batch")
    plt.ylabel("Loss (log scale)")
    plt.title("Loss Over Time (Batch)")

    # Customize ticks and grid
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.yscale("log")  # Use log scale for better visualization
    if batch_count > 20:
        plt.xticks(range(1, batch_count + 1, max(1, batch_count // 10)))
    plt.xlim([1, batch_count])

    # Adding legend
    plt.legend()

    # Generate status text
    if batch_count > 0:
        Final_loss = loss_history_Batches["combined"][-1]
        if Final_loss < 0.01:
            status_text = f"Batch {batch_count}: Exceptional performance! Loss is nearly zero ({Final_loss:.4f})."
        elif Final_loss < 0.05:
            status_text = f"Batch {batch_count}: Excellent progress! Minimal loss ({Final_loss:.4f})."
        elif Final_loss < 0.1:
            status_text = f"Batch {batch_count}: Great job! Loss is under control ({Final_loss:.4f})."
        elif Final_loss < 0.2:
            status_text = f"Batch {batch_count}: Good work, but there's room for improvement ({Final_loss:.4f})."
        elif Final_loss < 0.3:
            status_text = f"Batch {batch_count}: Moderate performance. Consider tuning the model or data ({Final_loss:.4f})."
        elif Final_loss < 0.5:
            status_text = f"Batch {batch_count}: Loss is slightly high. Try adjusting hyperparameters ({Final_loss:.4f})."
        else:
            status_text = f"Batch {batch_count}: High loss detected ({Final_loss:.4f}). Focus on debugging or re-evaluating the approach."
    else:
        status_text = "No data available for evaluation."

    # Add status text to the plot
    plt.text(0.5, 0.9, status_text, fontsize=12, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.7))

    # Save the plot
    base, ext = os.path.splitext(out_path)
    if ext.lower() not in [".png", ".jpg", ".jpeg", ".svg", ".pdf"]:
        out_path = base + ".png"

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print("Generated plot_loss_curves_Training_script_Batches diagram.")





def create_loss_diagrams(loss_history_Batches, loss_history_Epoches):
    train_logger.info("Creating Diagrams now")
    batches_figpath = os.path.join(diagramdirectory, "loss_curves_training_batches.png")
    epoch_figpath = os.path.join(diagramdirectory, "loss_curves_training_epoches.png")
    plot_loss_curves_Training_script_Batches(loss_history_Batches, out_path=batches_figpath)
    plot_loss_curves_Training_script_epoches(loss_history_Epoches, out_path=epoch_figpath)




#Logs spectrograms too tensorboard for visualisation
def log_spectrograms_to_tensorboard(audio, sr, tag, writer, global_step):
    stft = librosa.stft(audio)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max())

    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set_title(tag)
    fig.colorbar(ax.images[0], ax=ax)
    
    # Add the figure to TensorBoard
    writer.add_figure(tag, fig, global_step=global_step)
    plt.close(fig)







####FINE-TUNING#####
def plot_loss_curves_FineTuning_script_(loss_history_finetuning_epoches, out_path=r"C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_performance_logg\Diagrams\loss_curves_finetuning_epoches.png"):
    epochs_count = len(loss_history_finetuning_epoches["combined"])
    epochs = range(1, epochs_count + 1)
    plt.figure(figsize=(10,6))


    if len(loss_history_finetuning_epoches["l1"]) > 0:
        plt.plot(epochs, loss_history_finetuning_epoches["l1"], label="L1-loss", color="blue")


    if len(loss_history_finetuning_epoches["spectral"]) > 0:
        plt.plot(epochs, loss_history_finetuning_epoches["spectral"], label="spectral-loss", color="green")

    if len(loss_history_finetuning_epoches["perceptual"]) > 0:
        plt.plot(epochs, loss_history_finetuning_epoches["perceptual"], label="l1-loss", color="red")

    if len(loss_history_finetuning_epoches["mse"]) > 0:
        plt.plot(epochs, loss_history_finetuning_epoches["mse"], label="mse-loss", color="black")

    if len(loss_history_finetuning_epoches["multiscale"]) > 0:
        plt.plot(epochs, loss_history_finetuning_epoches["multiscale"], label="multiscale-loss", color="orange")



    plt.plot(epochs, loss_history_finetuning_epoches["combined"], label="combined-loss", color="purple")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Time (Batch)")
    plt.legend()


    Final_loss = loss_history_finetuning_epoches["combined"][-1] if len(loss_history_finetuning_epoches["combined"]) > 0 else 0
    if epochs_count  > 0:
        Final_loss = loss_history_finetuning_epoches["combined"][-1]
        if Final_loss < 0.01:
            status_text = f"Epoch {epochs}: Phenomenal ({Final_loss:.4f})"
        elif Final_loss < 0.05:
            status_text = f"Epoch {epochs}: Excellent (loss={Final_loss:.4f})"
        elif Final_loss < 0.1:
            status_text = f"Epoch {epochs}: Good (loss={Final_loss:.4f})"
        elif Final_loss < 0.2:
            status_text = f"Epoch {epochs}: Keep going (loss={Final_loss:.4f})"
        else:
            status_text = f"Epoch {epochs}: Higher loss ({Final_loss:.4f})"
    else:
        status_text = "No data"

    plt.text(0.5, 0.5, status_text, fontsize=12, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.7))


    base, ext = os.path.splitext(out_path)
    if ext.lower() not in [".png", ".jpg", ".jpeg", ".svg", ".pdf"]:
        out_path = base + ".png"

    plt.savefig(out_path)
    plt.close()
    print("Generated plot_loss_curves_FineTuning_script_ diagram.")



    






















