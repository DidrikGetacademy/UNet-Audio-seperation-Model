import torch
import deepspeed
import subprocess


# Check PyTorch and CUDA versions
print(torch.cuda.device_count())  # Skal vise 2
print(torch.cuda.get_device_name(0))  # RTX 3060
print(torch.cuda.get_device_name(1))  # GTX 1070
print("🔥 PyTorch Version:", torch.__version__)
print("🔧 CUDA Available:", torch.cuda.is_available())
print("🚀 CUDA Version in PyTorch:", torch.version.cuda)
print("🖥️ GPU Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("🎮 GPU Name:", torch.cuda.get_device_name(0))
    print("🎮 GPU Name:", torch.cuda.get_device_name(1))

# Check DeepSpeed version and capabilities
print("\n🚀 DeepSpeed Version:", deepspeed.__version__)
print("🛠️ CPUAdam Available:", "cpu_adam" in deepspeed.ops.__dict__)
print("🛠️ FusedAdam Available:", "fused_adam" in deepspeed.ops.__dict__)
print("🛠️ Transformer Kernel Available:", deepspeed.ops.op_builder.TransformerBuilder().is_compatible())

# Check CUDA Compiler version (NVCC)
try:
    nvcc_version = subprocess.check_output(["nvcc", "--version"]).decode("utf-8").strip().split("\n")[-1]
    print("\n🔧 NVCC Version:", nvcc_version)
except Exception as e:
    print("\n⚠️ NVCC not found or not installed in WSL.")

# Check NVIDIA driver and CUDA runtime
try:
    nvidia_smi_output = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
    print("\n🎮 NVIDIA-SMI Output:\n", nvidia_smi_output.split("\n")[2])  # Print the CUDA version from nvidia-smi
except Exception as e:
    print("\n⚠️ NVIDIA-SMI not found or not installed in WSL.")
