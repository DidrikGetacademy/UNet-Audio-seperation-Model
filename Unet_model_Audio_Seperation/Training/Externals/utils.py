# utils.py
import os
import platform

def Return_root_dir():
    if platform.system() == "Windows":
        root_dir = r"C:\Users\didri\Desktop\Programmering\Artifical intelligence Models\UNet-Models\Unet_model_Audio_Seperation"
    elif platform.system() == "Linux":
        root_dir = "/mnt/c/Users/didri/Desktop/Programmering/Artifical intelligence Models/UNet-Models/Unet_model_Audio_Seperation"
    else:
        raise OSError("Unsupported Platform")
    return root_dir
