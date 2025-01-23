import sys
import os
import torch
import matplotlib.pyplot as plt

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)  # Use insert(0) to prioritize this path

from Datasets.Scripts.Dataset_Musdb18 import MUSDB18StemDataset
from Model_Architecture.model import UNet
from Training.Externals.Functions import Return_root_dir

root_dir = Return_root_dir() #Gets the root directory

MUSDB18_dir = os.path.join(root_dir, "Datasets/Dataset_Audio_Folders/musdb18")

# Load the dataset
root_dir = MUSDB18_dir
print(f"Loading dataset from: {root_dir}")
dataset = MUSDB18StemDataset(root_dir=root_dir, subset="train")
print("Dataset loaded successfully.")

# Fetch one sample from the dataset
print("Fetching one sample from the dataset for visualization.")
mixture_tensor, vocals_tensor = dataset[0]
print("Sample fetched successfully.")
print(f"Shape of mixture tensor: {mixture_tensor.shape}")
print(f"Shape of vocals tensor: {vocals_tensor.shape}")

# Visualize the mixture spectrogram
print("Visualizing the mixture spectrogram.")
mixture_np = mixture_tensor.squeeze().numpy()
plt.imshow(mixture_np, aspect="auto", origin="lower", cmap="viridis")
plt.title("Mixture Spectrogram")
plt.colorbar()
plt.show()

# Reshape the tensor for batch dimension
print("Adding batch dimension to the mixture tensor.")
input_tensor = mixture_tensor.unsqueeze(0)
print(f"Input tensor shape (with batch dimension): {input_tensor.shape}")

# Initialize the model
print("Initializing the UNet model.")
model = UNet(in_channels=1, out_channels=1)

# Load model weights
model_path = os.path.join(root_dir,"Model_Weights/Onnx_model")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model weights not found at: {model_path}")

print(f"Loading model weights from: {model_path}")
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"),weights_only=True))
print("Model weights loaded successfully.")
model.eval()
print("Model set to evaluation mode.")

# Convert to ONNX
onnx_path = os.path.join(root_dir,"ONNX/model.onnx")
print(f"Saving the model to ONNX format at: {onnx_path}")

try:
    torch.onnx.export(
        model,
        input_tensor,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"}
        }
    )
    print(f"Model successfully converted to ONNX and saved at: {onnx_path}")
except Exception as e:
    print(f"Error during ONNX export: {e}")
