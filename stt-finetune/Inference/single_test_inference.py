import torch
import subprocess
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ------------------------------
# Configuration
# ------------------------------
MODEL_PATH = "../models/whisper_urdu_finetuned_v1"
AUDIO_FILE = "../training/data/Recording.m4a"
TARGET_SR = 16000

# ------------------------------
# FFmpeg loader (always works)
# ------------------------------
def load_audio_ffmpeg(path, target_sr=16000):
    cmd = [
        "ffmpeg",
        "-i", path,
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(target_sr),
        "-"
    ]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio = np.frombuffer(out.stdout, np.float32)
    return audio

# ------------------------------
# Load processor and model
# ------------------------------
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

# ------------------------------
# Load + Resample Audio
# ------------------------------
audio = load_audio_ffmpeg(AUDIO_FILE, TARGET_SR)

# ------------------------------
# Preprocess
# ------------------------------
inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
input_features = inputs.input_features.to(device)

# ------------------------------
# Generate
# ------------------------------
with torch.no_grad():
    predicted_ids = model.generate(
        input_features,
        num_beams=4,
        language="ur",
        task="transcribe"
    )

text = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

print("Predicted Roman transcription:")
print(text)
