import torch
import os
import sys

# Configure environment
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)  
from Training.Externals.Logger import setup_logger
from Training.Externals.utils import Return_root_dir

from Model_Architecture.model import UNet


root_dir = Return_root_dir() #Gets the root directory
train_log_path = os.path.join(root_dir, "Model_performance_logg/log/Model_Training_logg.txt")
Model_creation_logger = setup_logger('Model', train_log_path)

def create_and_save_model(input_shape, in_channels=1, out_channels=1, save_path="unet_vocal_isolation.pth"):

    # Select the device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Model_creation_logger.info(f"Using device: {device}")

    # Validate input shape
    if len(input_shape) != 4:
        raise ValueError("Input shape must be a 4-tuple: (batch_size, channels, height, width)")

    # Create a random input tensor on the selected device
    input_tensor = torch.randn(*input_shape, device=device)
    print("Random input tensor created.")

    print("Creating model...")

    # Initialize the U-Net model and move it to the selected device
    model = UNet(in_channels=in_channels, out_channels=out_channels).to(device)
    print("Model initialized.")

    # Print GPU memory usage if GPU is available
    if device.type == "cuda":
        Model_creation_logger.info("Allocated memory:", torch.cuda.memory_allocated() / 1024**2, "MB")
        Model_creation_logger.info("Max allocated memory:", torch.cuda.max_memory_allocated() / 1024**2, "MB")

    # Perform a forward pass
    try:
        output = model(input_tensor)
        Model_creation_logger.info("Forward pass successful.")
        Model_creation_logger.info(f"Input shape: {input_tensor.shape}")
        Model_creation_logger.info(f"Output shape: {output.shape}")
    except Exception as e:
        print("Error during forward pass:", str(e))
        return None

    # Save the model's state dictionary
    torch.save(model.state_dict(), save_path)
    print(f"Model state dictionary saved successfully at '{save_path}'.")


    full_model_path = save_path.replace(".pth", "_full.pth")
    torch.save(model, full_model_path)
    print(f"Entire model saved successfully at '{full_model_path}'.")

    return model

if __name__ == "__main__":

    input_shape = (1, 1, 256, 512) 


    create_and_save_model(input_shape, in_channels=1, out_channels=1, save_path="unet_vocal_isolation.pth")
