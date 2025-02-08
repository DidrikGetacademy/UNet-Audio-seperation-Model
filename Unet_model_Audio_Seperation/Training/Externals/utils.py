import os
import platform

def Return_root_dir():
    if platform.system() == "Windows":
        root_dir = r"C:\Users\didri\Desktop\Programmering\ArtificalintelligenceModels\UNet-Models\Unet_model_Audio_Seperation"
    elif platform.system() == "Linux":
        root_dir = "/mnt/c/Users/didri/Desktop/Programmering/ArtificalintelligenceModels/UNet-Model_Vocal_Isolation/Unet_model_Audio_Seperation"
    else:
        raise OSError("Unsupported Platform")
    return root_dir

