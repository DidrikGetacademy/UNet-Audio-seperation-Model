import onnxruntime as ort
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import sys
from scipy.io.wavfile import write
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
from Training.Externals.Functions import Return_root_dir

root_dir = Return_root_dir() #Gets the root directory

onnx_model_path = os.path.join(root_dir,"ONNX/model.onnx")
mixed_audio_path = os.path.join(root_dir,"Datasets/Dataset_Audio_Folders/Custom_Dataset/Input/withsound(1).wav")
vocals_audio_path = os.path.join(root_dir,"Datasets/Dataset_Audio_Folders/Custom_Dataset/Target/Only_vocals(1).WAV")

output_dir = os.path.join(root_dir,"ONNX/")

os.makedirs(output_dir, exist_ok=True)

# Load the ONNX model
if not os.path.exists(onnx_model_path):
    raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")

session = ort.InferenceSession(onnx_model_path)
session.set_providers(['CPUExecutionProvider'])  # Use CPU for inference

# Print input/output details
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"Input Name: {input_name}, Shape: {session.get_inputs()[0].shape}")
print(f"Output Name: {output_name}, Shape: {session.get_outputs()[0].shape}")

def load_audio(audio_path, sr=44100):
    """Load audio file and return the waveform and sampling rate."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at {audio_path}")
    print(f"\nLoading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=sr)
    return y, sr

def save_waveform(audio, sr, file_path, title="Waveform"):
    """Save the waveform as an image."""
    plt.figure(figsize=(10, 5))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.savefig(file_path)
    plt.close()
    print(f"Saved waveform: {file_path}")

def save_spectrogram(spectrogram, sr, file_path, title="Spectrogram"):
    """Save the spectrogram as an image."""
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), 
                             sr=sr, hop_length=512, x_axis='time', y_axis='log')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.savefig(file_path)
    plt.close()
    print(f"Saved spectrogram: {file_path}")


    
def reconstruct_audio_from_spectrogram(spectrogram, sr, hop_length=512):
    """Reconstruct audio from its spectrogram using iSTFT."""
    print("Reconstructing audio from spectrogram...")
    
    # Generate a random phase matrix initially
    phase = np.angle(librosa.stft(np.random.randn((spectrogram.shape[1] - 1) * hop_length), 
                                  n_fft=2048, hop_length=hop_length))

    # Ensure the phase and magnitude spectrogram match dimensions
    if spectrogram.shape != phase.shape:
        phase = phase[:, :spectrogram.shape[1]]  # Match time frames

    complex_spectrogram = spectrogram * np.exp(1j * phase)
    audio = librosa.istft(complex_spectrogram, hop_length=hop_length)
    return audio

# Process mixed audio
mixed_audio, sr = load_audio(mixed_audio_path)
mixed_spectrogram = np.abs(librosa.stft(mixed_audio, n_fft=2048, hop_length=512))

save_waveform(mixed_audio, sr, os.path.join(output_dir, "mixed_waveform.png"), "Mixed Audio Waveform")
save_spectrogram(mixed_spectrogram, sr, os.path.join(output_dir, "mixed_spectrogram.png"), "Mixed Audio Spectrogram")

# Process vocals-only audio
vocals_audio, _ = load_audio(vocals_audio_path)
vocals_spectrogram = np.abs(librosa.stft(vocals_audio, n_fft=2048, hop_length=512))

save_waveform(vocals_audio, sr, os.path.join(output_dir, "vocals_waveform.png"), "Vocals-Only Waveform")
save_spectrogram(vocals_spectrogram, sr, os.path.join(output_dir, "vocals_spectrogram.png"), "Vocals-Only Spectrogram")

# Prepare mixed audio spectrogram for inference
input_spectrogram = mixed_spectrogram.astype(np.float32)
input_spectrogram = np.expand_dims(input_spectrogram, axis=(0, 1))  # Add batch and channel dimensions
print(f"Input Spectrogram Shape for Inference: {input_spectrogram.shape}")

# Run inference on mixed audio
print("\nRunning ONNX inference on mixed audio...")
result = session.run([output_name], {input_name: input_spectrogram})
print("ONNX Inference Successful")

# Save the model's output spectrogram
output_spectrogram = np.squeeze(result[0])  # Remove batch and channel dimensions
save_spectrogram(output_spectrogram, sr, os.path.join(output_dir, "model_output_spectrogram.png"), "Model Output Spectrogram")

# Reconstruct and save audio from model output spectrogram
reconstructed_audio = reconstruct_audio_from_spectrogram(output_spectrogram, sr)
output_audio_path = os.path.join(output_dir, "model_output_audio.wav")
write(output_audio_path, sr, (reconstructed_audio * 32767).astype(np.int16))
print(f"Reconstructed audio saved at: {output_audio_path}")

print("\nAll visualizations and audio files processed and saved.")
