import torch
import deepspeed
import os

print("CUDA device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

print(torch.cuda.is_available())  # Should return True
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))  # Should print the name of your AMD GPU

def check_deepspeed_compatibility():
    # Check PyTorch version and CUDA
    print("Checking PyTorch and CUDA:")
    print("PyTorch version:", torch.__version__)
    print("CUDA version PyTorch uses:", torch.version.cuda)
    print("Is CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Current CUDA Device:", torch.cuda.get_device_name(torch.cuda.current_device()))
        print("CUDA device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA devices detected.")

    # Check DeepSpeed installation and environment compatibility
    print("\nChecking DeepSpeed: ")
    try:
        deepspeed_version = deepspeed.__version__
        print(f"DeepSpeed version: {deepspeed_version}")

        # Check DeepSpeed environment report
        print("\nRunning DeepSpeed environment report:")
        from deepspeed.utils import env_report
        deepspeed_report = env_report()
        print(deepspeed_report)

    except Exception as e:
        print("\nDeepSpeed installation check failed:")
        print(e)

    # Check environment variables for CUDA and CUTLASS
    print("\nChecking environment variables:")
    print("CUDA_HOME:", os.environ.get('CUDA_HOME'))
    print("CUTLASS_PATH:", os.environ.get('CUTLASS_PATH'))

if __name__ == "__main__":
    check_deepspeed_compatibility()
