import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Define paths
FAKE_DATA_PATH = "src/data/fakenews.csv"
TRUE_DATA_PATH = "src/data/truenews.csv"
PROCESSED_DATA_PATH = "src/data/processed_data.csv"

# Load datasets with error handling
try:
    df_fake = pd.read_csv(FAKE_DATA_PATH)
    df_true = pd.read_csv(TRUE_DATA_PATH)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Verify required columns exist
required_columns = ["text"]
for df, name in [(df_fake, "Fake News"), (df_true, "True News")]:
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Missing '{col}' column in {name} dataset.")
            exit()

# Add Labels (0 = True, 1 = Fake)
df_fake["label"] = 1
df_true["label"] = 0

# Combine datasets
df = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)

# Save processed data
df.to_csv(PROCESSED_DATA_PATH, index=False)

# Display dataset info
print("Dataset Overview:")
print(df.info())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Class Distribution Plot
plt.figure(figsize=(6, 4))
sns.countplot(x=df['label'], hue=df['label'], palette=["blue", "red"], legend=False)
plt.title("Class Distribution (Real vs. Fake)")
plt.xlabel("Label (0 = Real, 1 = Fake)")
plt.ylabel("Count")
plt.show()

# Generate Word Clouds
fake_text = " ".join(df[df["label"] == 1]["text"])
real_text = " ".join(df[df["label"] == 0]["text"])

wordcloud_fake = WordCloud(width=800, height=400, background_color="black").generate(fake_text)
wordcloud_real = WordCloud(width=800, height=400, background_color="white").generate(real_text)

# Fake News Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_fake, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - Fake News")
plt.show()

# Real News Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_real, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - Real News")
plt.show()
