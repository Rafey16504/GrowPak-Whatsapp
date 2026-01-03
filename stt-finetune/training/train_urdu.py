import os
import subprocess
import numpy as np
import torch
import pandas as pd
from datasets import load_from_disk
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# ------------------------------
# Configuration  
# ------------------------------
MODEL_NAME = "openai/whisper-small"
DATASET_PATH = "urdu_asr_dataset"
TARGET_SR = 16000
CACHE_DIR = "audio_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

MAX_LABEL_TOKENS = 448  # Whisper training limit

# ------------------------------
# FFmpeg audio loader + caching
# ------------------------------
def ffmpeg_load(file_path, target_sr=TARGET_SR):
    cache_file = os.path.join(CACHE_DIR, os.path.basename(file_path) + ".npy")

    if os.path.exists(cache_file):
        return np.load(cache_file)

    command = [
        "ffmpeg",
        "-i", file_path,
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(target_sr),
        "-"
    ]
    out = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    audio = np.frombuffer(out.stdout, dtype=np.float32)

    np.save(cache_file, audio)
    return audio


# ------------------------------
# Data collator
# ------------------------------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

# =============================================================
# ADD: Training Logger Callback → outputs training_logs.csv
# =============================================================
class CSVLoggerCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logs["step"] = state.global_step
            logs["epoch"] = state.epoch
            self.logs.append(logs)

    def on_train_end(self, args, state, control, **kwargs):
        df = pd.DataFrame(self.logs)
        save_path = os.path.join(self.output_dir, "training_logs.csv")
        df.to_csv(save_path, index=False)
        print(f"\n✅ Training logs saved to: {save_path}\n")
        
# ------------------------------
# Main training
# ------------------------------
def main():
    print("Loading dataset...")
    dataset = load_from_disk(DATASET_PATH)

    print("Loading model + processor...")
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME,
        language="ur",             
        task="transcribe" 
    )

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Freeze encoder for faster training
    # for param in model.model.encoder.parameters():
    #     param.requires_grad = False

    # Force decoder language = Urdu
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ur", task="transcribe")
    # model.config.suppress_tokens = []   # no character blocking

    # if processor.tokenizer.pad_token is None:
    #     processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # use_bf16 = torch.cuda.is_available()
    # if not use_bf16:
    #     print("CUDA not available — running on CPU/no bf16.")

    skipped_long = 0

    # --------------------------
    # Preprocess function
    # --------------------------
    def preprocess(batch):
        nonlocal skipped_long

        audio_field = batch["audio"]
        path = audio_field["path"] if isinstance(audio_field, dict) else audio_field

        waveform = ffmpeg_load(path)
        inputs = processor.feature_extractor(waveform, sampling_rate=TARGET_SR)
        batch["input_features"] = inputs.input_features[0]

        text = batch.get("text") or ""
        enc = processor.tokenizer(text_target=text)
        ids = enc.input_ids

        if len(ids) > MAX_LABEL_TOKENS:
            skipped_long += 1
            batch["labels"] = []  # placeholder
            batch["skip"] = True
        else:
            batch["labels"] = ids
            batch["skip"] = False

        return batch

    # Columns to remove
    cols_keep = {"text", "audio", "file_path"}
    cols_to_remove = [c for c in dataset["train"].column_names if c not in cols_keep]

    # Apply map
    dataset = dataset.map(
        preprocess,
        remove_columns=cols_to_remove,
        num_proc=1,
    )

    # Filter skipped
    before = len(dataset["train"])
    dataset["train"] = dataset["train"].filter(lambda x: not x["skip"])
    after = len(dataset["train"])

    print(f"Skipped long samples (>448 tokens): {skipped_long}")
    print(f"Train set reduced: {before} → {after}")

    # Remove helper
    dataset = dataset.remove_columns(["skip"])

    # --------------------------
    # Training args
    # --------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir="../models/whisper_urdu_finetuned_v1.1",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=100,
        max_steps=1000,
        eval_steps=100,
        logging_steps=50,
        fp16=False,
        bf16=True,
        predict_with_generate=True,
        generation_max_length=225,
        # push_to_hub=False,
    )

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

    # --------------------------
    # Trainer
    # --------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[CSVLoggerCallback("../models/whisper_urdu_finetuned_v1.1")],
    )

    # Train
    print("Starting fine-tuning…")
    trainer.train()

    # Save model + processor so check.py loads correctly
    trainer.save_model("../models/whisper_urdu_finetuned_v1.1")
    processor.save_pretrained("../models/whisper_urdu_finetuned_v1.1")

    print("Training complete!")


if __name__ == "__main__":
    main()
