import os
import pandas as pd
from collections import Counter
import shutil
from tqdm import tqdm

# Paths
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "train.csv")
# Where images are extracted (e.g., train/0/1/2/0123456789abcdef.jpg)
IMAGES_ROOT = "train"
OUTPUT_ROOT = os.path.join(DATA_DIR, "train")
NUM_CLASSES = 5
IMAGES_PER_CLASS = 100  # Limit per class for prototyping

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Download train.csv if not present
if not os.path.exists(CSV_PATH):
    import urllib.request
    print("Downloading train.csv...")
    urllib.request.urlretrieve(
        "https://s3.amazonaws.com/google-landmark/metadata/train.csv", CSV_PATH)

# Read CSV
print("Reading train.csv...")
df = pd.read_csv(CSV_PATH)

# Scan for available images in the extracted train/ directory
print("Scanning for available images in extracted train/ directory...")
available_image_ids = []
for root, dirs, files in os.walk(IMAGES_ROOT):
    for file in files:
        if file.endswith('.jpg'):
            img_id = file[:-4]  # remove .jpg
            available_image_ids.append(img_id)

print(f"Found {len(available_image_ids)} available images.")

# Filter dataframe to only available images
df_available = df[df['id'].isin(available_image_ids)]

# Find top N landmark_ids among available images
landmark_counts = Counter(df_available['landmark_id'])
top_landmarks = [lid for lid, _ in landmark_counts.most_common(NUM_CLASSES)]

print(f"Top {NUM_CLASSES} landmark_ids among available images: {top_landmarks}")

# Filter for top classes
df_top = df_available[df_available['landmark_id'].isin(top_landmarks)]

# Organize images
for landmark_id in top_landmarks:
    class_dir = os.path.join(OUTPUT_ROOT, str(landmark_id))
    os.makedirs(class_dir, exist_ok=True)
    images = df_top[df_top['landmark_id'] == landmark_id]['id'].tolist()[
        :IMAGES_PER_CLASS]
    for img_id in tqdm(images, desc=f"Copying for landmark {landmark_id}"):
        # Images are in train/a/b/c/id.jpg
        a, b, c = img_id[0], img_id[1], img_id[2]
        src = os.path.join(IMAGES_ROOT, a, b, c, f"{img_id}.jpg")
        dst = os.path.join(class_dir, f"{img_id}.jpg")
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {src} not found.")
print("Done organizing images.")
