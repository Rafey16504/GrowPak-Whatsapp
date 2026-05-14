import pandas as pd
import numpy as np

# ----------------------------------------------------
# Load excel file
# ----------------------------------------------------
df = pd.read_excel("model_accuracy_v1.1(urdu).xlsx")

# ----------------------------------------------------
# Drop rows where Corrected Language is missing
# ----------------------------------------------------
df = df[df["Corrected Language"].notna()]
df["Corrected Language"] = df["Corrected Language"].str.strip()

# Keep only valid languages
df = df[df["Corrected Language"].isin(["ur", "pa"])]

# ----------------------------------------------------
# Cleaning function
# ----------------------------------------------------
def clean(x):
    if isinstance(x, str):
        return x.strip().replace(" ", "").replace("‌", "")
    return ""

df["ground_truth"] = df["ground_truth"].apply(clean)
df["prediction"] = df["prediction"].apply(clean)
df["auto_text_urdu"] = df["auto_text_urdu"].apply(clean)

# ----------------------------------------------------
# Levenshtein distance
# ----------------------------------------------------
def levenshtein(a, b):
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
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[m][n]

# ----------------------------------------------------
# WER
# ----------------------------------------------------
def wer(gt, pred):
    gt_words = gt.split()
    pred_words = pred.split()
    return levenshtein(gt_words, pred_words) / max(len(gt_words), 1)

# ----------------------------------------------------
# Reload raw text for WER
# ----------------------------------------------------
df_raw = pd.read_excel("model_accuracy_v1.1(urdu).xlsx")
df_raw = df_raw.loc[df.index]

df["WER_pred"] = df.apply(
    lambda r: wer(str(df_raw.loc[r.name, "ground_truth"]),
                  str(df_raw.loc[r.name, "prediction"])),
    axis=1
)

df["WER_auto"] = df.apply(
    lambda r: wer(str(df_raw.loc[r.name, "ground_truth"]),
                  str(df_raw.loc[r.name, "auto_text_urdu"])),
    axis=1
)

# ----------------------------------------------------
# Language-wise report
# ----------------------------------------------------
print("========== LANGUAGE-WISE REPORT ==========")

for lang in ["ur", "pa"]:
    sub = df[df["Corrected Language"] == lang]

    if len(sub) == 0:
        continue

    exact_pred_acc = (sub["prediction"] == sub["ground_truth"]).mean()
    exact_auto_acc = (sub["auto_text_urdu"] == sub["ground_truth"]).mean()

    wer_pred = sub["WER_pred"].mean()
    wer_auto = sub["WER_auto"].mean()

    print(f"\nLanguage: {lang}")
    print("-------------------------------------")
    print(f"Samples:                    {len(sub)}")
    print(f"Prediction WER:             {wer_pred*100:.4f}")
    print(f"Auto WER:                   {wer_auto*100:.4f}")
    print(f"Prediction Accuracy (WER):  {(1 - wer_pred)*100:.4f}%")
    print(f"Auto Accuracy (WER):        {(1 - wer_auto)*100:.4f}%")
    print(f"Prediction Exact Match:     {exact_pred_acc*100:.4f}%")
    print(f"Auto Exact Match:           {exact_auto_acc*100:.4f}%")
