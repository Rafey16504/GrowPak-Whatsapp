import pandas as pd
import os
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

EXCEL_PATH = "transcripts_new.xlsx"
AUDIO_FOLDER = "data/audio"

def load_dataset():
    df = pd.read_excel(EXCEL_PATH)

    # Keep only rows with valid transcription
    df = df[["file_name", "urdu_script", "Corrected Language"]]
    df = df.rename(columns={"urdu_script": "text",
                            "Corrected Language": "language"})

    # Construct full audio paths
    df["audio"] = df["file_name"].apply(
        lambda x: os.path.join(AUDIO_FOLDER, x)
    )

    # Remove rows with missing files
    df = df[df["audio"].apply(os.path.exists)]

    # Split train/validation
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    return DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
    })

if __name__ == "__main__":
    dset = load_dataset()
    dset.save_to_disk("urdu_asr_dataset")
    print("Dataset saved.")
