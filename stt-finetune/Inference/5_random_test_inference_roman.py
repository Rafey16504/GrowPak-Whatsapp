import os
import random
import torch
import pandas as pd
from scipy.signal import resample
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
MODEL_PATH = "../models/whisper_roman_finetuned_v1.1"   # your finetuned model folder
AUDIO_FOLDER = "../training/data/audio"
EXCEL_FILE = "../training/transcripts_new.xlsx"

AUDIO_COLUMN = "file_name"
TEXT_COLUMN = "Corrected Transcription"
NUM_SAMPLES = 5

# ----------------------------------------------------
# LOAD MODEL + PROCESSOR (IMPORTANT: load from finetuned model)
# ----------------------------------------------------
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)

# Make sure no forced decoder IDs (matches your finetuning setup)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


def resample_audio(audio, orig_sr, target_sr=16000):
    if orig_sr == target_sr:
        return audio
    duration = audio.shape[0] / orig_sr
    target_length = int(duration * target_sr)
    return resample(audio, target_length)
# ----------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------
df = pd.read_excel(EXCEL_FILE)
df[AUDIO_COLUMN] = df[AUDIO_COLUMN].astype(str)

# ----------------------------------------------------
# FIND AUDIO FILES
# ----------------------------------------------------
all_audio_files = [
    f for f in os.listdir(AUDIO_FOLDER)
    if f.lower().endswith(".opus")
]

df = df[df[AUDIO_COLUMN].isin(all_audio_files)]
samples = df.sample(NUM_SAMPLES)

# ----------------------------------------------------
# RUN EVALUATION
# ----------------------------------------------------
for idx, row in samples.iterrows():
    file_name = row[AUDIO_COLUMN]
    gt_text = row[TEXT_COLUMN]
    audio_path = os.path.join(AUDIO_FOLDER, file_name)

    print("\n----------------------------------------------------")
    print(f"FILE: {file_name}")

    # Load audio
    audio, sr = sf.read(audio_path)
    audio = resample_audio(audio, sr, 16000)
    # Preprocess (uses your finetuned preprocessor_config + normalizer)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    # Predict
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            num_beams=1,
        )

    predicted_text = processor.tokenizer.decode(
        predicted_ids[0],
        skip_special_tokens=True
    )

    # Output result
    print(f"ACTUAL:    {gt_text}")
    print(f"PREDICTED: {predicted_text}")
