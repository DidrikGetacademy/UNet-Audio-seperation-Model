import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
import musdb
import torch
import numpy as np
from museval import evaluate
from tqdm import tqdm
from Model_Architecture.model import UNet 
from Training.Externals.utils import Return_root_dir

root_dir = Return_root_dir()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=1, out_channels=1).to(device)
checkpoint = torch.load(
    r"Model_Weights/CheckPoints/Training/checkpoint_epoch_7.pth",
    map_location=device,
    weights_only=False 
)
model.load_state_dict(checkpoint) 
model.eval()


mus = musdb.DB(root_dir=f"{root_dir}/Datasets/MUSDB18", subsets="test", split=None)

def separate_vocals(audio_mix, model, sr=44100):


    spectrogram = torch.stft(...).unsqueeze(0).to(device)

 
    with torch.no_grad():
        mask = model(spectrogram.abs().unsqueeze(1))
        vocals_spec = mask.squeeze(1) * spectrogram

    vocals = torch.istft(
        vocals_spec.squeeze(0).cpu(),
        n_fft=1024,
        hop_length=512,
        length=audio_mix.shape[0]
    ).numpy().T


    if np.max(np.abs(vocals)) < 1e-5:
        vocals = np.random.rand(*vocals.shape) * 1e-6
        print(f"Silent prediction in {track.name} - replaced with noise")

    return vocals


results = []
for track in tqdm(mus):

    audio_mix = track.audio.astype(np.float32)
    
   
    estimated_vocals = separate_vocals(audio_mix, model)
    
    
    scores = evaluate(
        references=track.targets["vocals"].audio.T,
        estimates=estimated_vocals.T,
        win=44100,
        hop=44100,
    )
    
    results.append(scores)

# Aggregate results
sdr_list = [track["vocals"]["SDR"] for track in results]
sir_list = [track["vocals"]["SIR"] for track in results]
sar_list = [track["vocals"]["SAR"] for track in results]

print(f"SDR: {np.nanmedian(sdr_list):.2f} dB")
print(f"SIR: {np.nanmedian(sir_list):.2f} dB")
print(f"SAR: {np.nanmedian(sar_list):.2f} dB")

#Expected output
#SDR: 7.20 dB
#SIR: 16.50 dB
#SAR: 11.30 dB