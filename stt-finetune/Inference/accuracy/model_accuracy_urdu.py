import pandas as pd
import numpy as np

# ----------------------------------------------------
# Load your excel file
# ----------------------------------------------------
df = pd.read_excel("model_accuracy_v1.1(urdu).xlsx")

# ----------------------------------------------------
# Cleaning function
# ----------------------------------------------------
def clean(x):
    if isinstance(x, str):
        # Remove spaces + zero-width joiners
        return x.strip().replace(" ", "").replace("‌", "")
    return ""

df["ground_truth"] = df["ground_truth"].apply(clean)
df["prediction"] = df["prediction"].apply(clean)
df["auto_text_urdu"] = df["auto_text_urdu"].apply(clean)

# ----------------------------------------------------
# Levenshtein distance
# ----------------------------------------------------
def levenshtein(a, b):
    """Compute Levenshtein edit distance between two strings."""
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[m][n]

# ----------------------------------------------------
# WER = edit_distance(words) / number_of_words_in_ground_truth
# ----------------------------------------------------
def wer(gt, pred):
    gt_words = gt.split()
    pred_words = pred.split()
    return levenshtein(gt_words, pred_words) / max(len(gt_words), 1)

# ----------------------------------------------------
# Compute accuracy (% exact match)
# ----------------------------------------------------
pred_acc = (df["prediction"] == df["ground_truth"]).mean()
auto_acc = (df["auto_text_urdu"] == df["ground_truth"]).mean()

# For WER we need spaces, so use non-cleaned versions for WER.
# Load original again but only for WER usage:
df_raw = pd.read_excel("model_accuracy_v1.1(urdu).xlsx")

df["WER_pred"] = df.apply(
    lambda r: wer(str(df_raw.loc[r.name, "ground_truth"]), 
                  str(df_raw.loc[r.name, "prediction"])), axis=1)

df["WER_auto"] = df.apply(
    lambda r: wer(str(df_raw.loc[r.name, "ground_truth"]), 
                  str(df_raw.loc[r.name, "auto_text_urdu"])), axis=1)

# ----------------------------------------------------
# Print report
# ----------------------------------------------------


print(f"Prediction WER:            {df['WER_pred'].mean()*100:.4f}")
print(f"Auto WER:                  {df['WER_auto'].mean()*100:.4f}")
print("=====================================")

print("========== ACCURACY REPORT ==========")
print(f"Prediction Accuracy:       {(1 - df['WER_pred'].mean())*100:.4f}%")
print(f"Urdu Auto Accuracy:        {(1 - df['WER_auto'].mean())*100:.4f}%")