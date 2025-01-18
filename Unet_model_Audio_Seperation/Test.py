import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"CUDA is available! Device name: {torch.cuda.get_device_name(0)}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
else:
    print("CUDA is not available.")
