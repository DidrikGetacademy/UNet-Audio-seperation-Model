import torch

# Check PyTorch version
print('PyTorch Version:', torch.__version__)

# Check if CUDA is available
print('CUDA Available:', torch.cuda.is_available())

if torch.cuda.is_available():
    # Get GPU details
    print('GPU Name:', torch.cuda.get_device_name(0))
    print('Number of GPUs:', torch.cuda.device_count())
    print('Current GPU Device:', torch.cuda.current_device())
