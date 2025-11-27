# helper function to download and format the dataset for the YOLO model to train off of.

import kagglehub
import os
import shutil
import glob
import random
from tqdm import tqdm  # pip install tqdm

def move_files(file_list, split_name):
    for img_src, txt_src in tqdm(file_list):
        f_name = os.path.basename(img_src)
        l_name = os.path.basename(txt_src)
        
        # MOVE Image
        shutil.move(img_src, f"{BASE_DIR}/train/images/{split_name}/{f_name}")
        # MOVE Label
        shutil.move(txt_src, f"{BASE_DIR}/train/labels/{split_name}/{l_name}")

if __name__ == "__main__":

    path = kagglehub.dataset_download("sakshamjn/vehicle-detection-8-classes-object-detection")

    images = os.path.join(path, 'train', 'images')
    labels = os.path.join(path, 'train', 'labels')

    BASE_DIR = "data" 

    for split in ['train', 'val']:
        os.makedirs(f"{BASE_DIR}/train/images/{split}", exist_ok=True)
        os.makedirs(f"{BASE_DIR}/train/labels/{split}", exist_ok=True)

    all_images = glob.glob(os.path.join(images, "*"))
    pairs = []

    for img_path in all_images:
        filename = os.path.basename(img_path)
        filename_no_ext = os.path.splitext(filename)[0]
        txt_path = os.path.join(labels, filename_no_ext + ".txt")
        if os.path.exists(txt_path):
            pairs.append((img_path, txt_path))


    random.seed(42)
    random.shuffle(pairs)

    split_idx = int(len(pairs) * 0.8)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    move_files(train_pairs, 'train')
    move_files(val_pairs, 'val')

    # Downloading and moving video file
    video_path = kagglehub.dataset_download("chicicecream/720p-road-and-traffic-video-for-object-detection")

    print("Path to dataset files:", video_path)

    TEST_DIR = f"{BASE_DIR}/test"
    os.makedirs(f"{BASE_DIR}/test")

    videos = glob.glob(os.path.join(video_path, "**", "*.mp4"), recursive=True)

    if videos:

        vid_src = videos[0]

        dest_path = os.path.join(TEST_DIR, "test_video.mp4")
        
        shutil.move(vid_src, dest_path)

    