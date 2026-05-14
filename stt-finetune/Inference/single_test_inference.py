import torch
import subprocess
import numpy as np
from pathlib import Path
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
)

# ------------------------------
# Configuration
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (BASE_DIR / ".." / "models" / "whisper_urdu_finetuned_v2").resolve()
AUDIO_FILE = (BASE_DIR / ".." / "training" / "asr_dataset" / "data" / "Recording.m4a").resolve()
TARGET_SR = 16000

# ------------------------------
# FFmpeg loader (always works)
# ------------------------------
def load_audio_ffmpeg(path, target_sr=16000):
    cmd = [
        "ffmpeg",
        "-i", str(path),
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(target_sr),
        "-"
    ]
    out = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    audio = np.frombuffer(out.stdout, np.float32)
    if audio.size == 0:
        raise RuntimeError(
            f"No audio decoded from {path}. FFmpeg output:\n{out.stderr.decode(errors='ignore')}"
        )
    return audio


def load_whisper_processor(model_path: Path) -> WhisperProcessor:
    preprocessor_config = model_path / "preprocessor_config.json"
    if preprocessor_config.exists():
        return WhisperProcessor.from_pretrained(str(model_path))

    # Backward-compatible fallback for checkpoints saved without preprocessor_config.json
    feature_extractor = WhisperFeatureExtractor(
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )
    tokenizer = WhisperTokenizer.from_pretrained(str(model_path))
    return WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# ------------------------------
# Load processor and model
# ------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")
if not AUDIO_FILE.exists():
    raise FileNotFoundError(f"Audio file not found: {AUDIO_FILE}")

processor = load_whisper_processor(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(str(MODEL_PATH))

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
input_features = inputs.input_features.to(device=device, dtype=next(model.parameters()).dtype)

# ------------------------------
# Generate
# ------------------------------
with torch.no_grad():
    predicted_ids = model.generate(
        input_features,
        num_beams=2,
        language="ur",
        task="transcribe"
    )

text = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

print("Predicted Urdu transcription:")
print(text)
