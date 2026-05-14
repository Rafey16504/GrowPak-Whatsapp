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
AUDIO_FOLDER = "../training/data/audio_second_batch"
EXCEL_FILE = "../training/second_batch.xlsx"

AUDIO_COLUMN = "file_name"

OUTPUT_XLSX = "second_batch_transcripts_urdu.xlsx"

SAMPLE_RATE = 16000
CHUNK_LENGTH_SEC = 30
CHUNK_OVERLAP_SEC = 2

# ----------------------------------------------------
# LOAD MODEL + PROCESSOR
# ----------------------------------------------------
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# ----------------------------------------------------
# AUDIO HELPERS
# ----------------------------------------------------
def resample_audio(audio, orig_sr, target_sr=16000):
    if orig_sr == target_sr:
        return audio
    duration = audio.shape[0] / orig_sr
    target_length = int(duration * target_sr)
    return resample(audio, target_length)


def chunk_audio(audio, sr, chunk_sec=30, overlap_sec=2):
    chunk_size = int(chunk_sec * sr)
    overlap_size = int(overlap_sec * sr)
    step = chunk_size - overlap_size

    chunks = []
    for start in range(0, len(audio), step):
        end = start + chunk_size
        chunk = audio[start:end]
        if len(chunk) < sr:  # skip extremely small tail
            continue
        chunks.append(chunk)

    return chunks

def merge_transcripts(prev, curr, max_overlap_words=25):
    """
    Removes duplicated overlap between two consecutive chunk transcripts
    """
    prev_words = prev.split()
    curr_words = curr.split()

    max_check = min(len(prev_words), len(curr_words), max_overlap_words)

    for i in range(max_check, 0, -1):
        if prev_words[-i:] == curr_words[:i]:
            return prev + " " + " ".join(curr_words[i:])

    return prev + " " + curr

def transcribe_audio_chunks(audio):
    chunks = chunk_audio(
        audio,
        SAMPLE_RATE,
        CHUNK_LENGTH_SEC,
        CHUNK_OVERLAP_SEC
    )

    merged_text = ""

    for i, chunk in enumerate(chunks):
        inputs = processor(
            chunk,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                num_beams=4,                
                repetition_penalty=1.15,    # reduces repetition
                no_repeat_ngram_size=3       # prevents loops 
            )

        text = processor.tokenizer.decode(
            predicted_ids[0],
            skip_special_tokens=True
        ).strip()

        if not text:
            continue

        if not merged_text:
            merged_text = text
        else:
            merged_text = merge_transcripts(merged_text, text)

    return merged_text



# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------
df = pd.read_excel(EXCEL_FILE)
df[AUDIO_COLUMN] = df[AUDIO_COLUMN].astype(str)

results = []

# ----------------------------------------------------
# RUN INFERENCE (FULL AUDIO)
# ----------------------------------------------------
for idx, row in df.iterrows():
    file_name = row[AUDIO_COLUMN]
    audio_path = os.path.join(AUDIO_FOLDER, file_name)

    print(f"\nProcessing: {file_name}")

    if not os.path.exists(audio_path):
        print("  -> Missing audio file.")
        results.append([file_name, None, "missing_audio"])
        continue

    try:
        # Load audio
        audio, sr = sf.read(audio_path)

        # Convert stereo → mono if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample
        audio = resample_audio(audio, sr, SAMPLE_RATE)

        # Transcribe full audio (chunked)
        predicted_text = transcribe_audio_chunks(audio)

        print(f"  PR: {predicted_text}")

        results.append([file_name, predicted_text, "ok"])

    except Exception as e:
        print(f"  ERROR: {e}")
        results.append([file_name, None, "error"])

# ----------------------------------------------------
# SAVE XLSX
# ----------------------------------------------------
out_df = pd.DataFrame(
    results,
    columns=["file_name", "prediction", "status"]
)

out_dir = os.path.dirname(OUTPUT_XLSX)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

out_df.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")

print("\n----------------------------------------------------")
print(f"Finished! Saved results to {OUTPUT_XLSX}")
print("----------------------------------------------------")