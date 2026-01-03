import os
import torch
import pandas as pd
import soundfile as sf
from scipy.signal import resample
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
MODEL_PATH = "../models/whisper_urdu_finetuned_v1.1"
AUDIO_FOLDER = "../training/data/audio"
EXCEL_FILE = "../training/transcripts_new.xlsx"

AUDIO_COLUMN = "file_name"
TEXT_COLUMN = "auto_text_urdu"
GT_TEXT_COLUMN = "urdu_script"
OUTPUT_XLSX = "accuracy/model_accuracy_v1.1(urdu).xlsx"

# ----------------------------------------------------
# LOAD MODEL + PROCESSOR
# ----------------------------------------------------
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)

forced_ids = processor.get_decoder_prompt_ids(language="ur", task="transcribe")

model.generation_config.forced_decoder_ids = forced_ids
model.generation_config.language = "ur"
model.generation_config.task = "transcribe"
model.generation_config.suppress_tokens = []

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# ----------------------------------------------------
# RESAMPLING FUNCTION
# ----------------------------------------------------
def resample_audio(audio, orig_sr, target_sr=16000):
    if orig_sr == target_sr:
        return audio
    duration = audio.shape[0] / orig_sr
    target_length = int(duration * target_sr)
    return resample(audio, target_length)

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------
df = pd.read_excel(EXCEL_FILE)
df[AUDIO_COLUMN] = df[AUDIO_COLUMN].astype(str)

results = []

# ----------------------------------------------------
# RUN INFERENCE ON ENTIRE DATASET
# ----------------------------------------------------
for idx, row in df.iterrows():
    file_name = row[AUDIO_COLUMN]
    gt_text = row[GT_TEXT_COLUMN]     # ground truth
    auto_text_urdu = row[TEXT_COLUMN] # additional copy

    audio_path = os.path.join(AUDIO_FOLDER, file_name)

    print(f"\nProcessing: {file_name}")

    if not os.path.exists(audio_path):
        print("  -> Missing audio file.")
        results.append([file_name, gt_text, None, auto_text_urdu, "missing_audio"])
        continue

    try:
        # Load audio
        audio, sr = sf.read(audio_path)

        # Resample to 16k
        audio = resample_audio(audio, sr, 16000)

        # Preprocess
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        # Predict
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                num_beams=4,
                language="ur",
    task="transcribe"
            )

        predicted_text = processor.tokenizer.decode(
            predicted_ids[0], skip_special_tokens=True
        )

        print(f"  GT:  {gt_text}")
        print(f"  PR:  {predicted_text}")

        results.append([file_name, gt_text, predicted_text, auto_text_urdu, "ok"])

    except Exception as e:
        print(f"  ERROR: {e}")
        results.append([file_name, gt_text, None, auto_text_urdu, "error"])

# ----------------------------------------------------
# SAVE XLSX
# ----------------------------------------------------
out_df = pd.DataFrame(
    results,
    columns=["file_name", "ground_truth", "prediction", "auto_text_urdu", "status"]
)

out_df.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")

print("\n----------------------------------------------------")
print(f"Finished! Saved results to {OUTPUT_XLSX}")
print("----------------------------------------------------")
