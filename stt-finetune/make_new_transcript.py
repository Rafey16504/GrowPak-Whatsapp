import pandas as pd

# Load files
old = pd.read_excel("transcripts_old.xlsx")
corr = pd.read_excel("corrected_roman_with_urdu.xlsx")

# Remove unnamed columns
old = old.loc[:, ~old.columns.str.contains('^Unnamed')]

# Create lookup dictionaries
roman_lookup = dict(zip(corr["file_name"], corr["roman_urdu"]))
urdu_lookup = dict(zip(corr["file_name"], corr["urdu_script"]))

# Replace Corrected Transcription
old["Corrected Transcription"] = old["file_name"].map(roman_lookup)

# Add urdu_script at the end
old["urdu_script"] = old["file_name"].map(urdu_lookup)

# Save final file
old.to_excel("transcripts_new.xlsx", index=False)

print("Done! File saved as transcripts_new.xlsx")
