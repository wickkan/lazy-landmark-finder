import pandas as pd
import os

# Paths
csv_path = "data/train.csv"
images_root = "train"
filtered_csv_path = "data/train_000.csv"

# Read CSV
df = pd.read_csv(csv_path)

# Only keep rows where the image file exists (i.e., starts with '0')


def image_exists(img_id):
    img_path = os.path.join(
        images_root, img_id[0], img_id[1], img_id[2], f"{img_id}.jpg")
    return os.path.exists(img_path)


filtered_df = df[df['id'].str.startswith(
    '0') & df['id'].apply(image_exists)].copy()

# Remap landmark_id to contiguous range
unique_landmarks = sorted(filtered_df['landmark_id'].unique())
landmark2idx = {landmark: idx for idx, landmark in enumerate(unique_landmarks)}
filtered_df['landmark_id'] = filtered_df['landmark_id'].map(landmark2idx)

filtered_df.to_csv(filtered_csv_path, index=False)
print(
    f"Filtered and remapped CSV saved to {filtered_csv_path} with {len(filtered_df)} rows and {len(unique_landmarks)} classes.")
