import os
import hashlib
import librosa
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def get_audio_hash(file_path):
    y, sr = librosa.load(file_path, sr=48000)
    y_bytes = y.tobytes()
    return hashlib.sha256(y_bytes).hexdigest()

def collect_hashes(folder_path):
    hash_dict = defaultdict(list)
    for fname in os.listdir(folder_path):
        if fname.endswith(".wav"):
            full_path = os.path.join(folder_path, fname)
            try:
                h = get_audio_hash(full_path)
                hash_dict[h].append(fname)
            except:
                print(f"❌ Failed to read: {fname}")
    return hash_dict

def plot_distribution(data, title):
    keys = list(data.keys())
    values = [len(data[k]) for k in keys]
    plt.bar(keys, values)
    plt.title(title)
    plt.xlabel("Hash")
    plt.ylabel("Number of files")
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    plt.show()

def detect_duplicates(train_hashes, test_hashes, val_hashes):
    print("\n📊 Checking for duplicates between sets...")
    train_set = set(train_hashes.keys())
    test_set = set(test_hashes.keys())
    val_set = set(val_hashes.keys())

    dup_train_test = train_set.intersection(test_set)
    dup_train_val = train_set.intersection(val_set)
    dup_val_test = val_set.intersection(test_set)

    if dup_train_test:
        print(f"❗ Train/Test overlap: {len(dup_train_test)} files")
    if dup_train_val:
        print(f"❗ Train/Val overlap: {len(dup_train_val)} files")
    if dup_val_test:
        print(f"❗ Val/Test overlap: {len(dup_val_test)} files")

    if not (dup_train_test or dup_train_val or dup_val_test):
        print("✅ No duplicates between datasets!")

def run_check(train_dir, val_dir, test_dir):
    print("🔍 Collecting hashes from train set...")
    train_hashes = collect_hashes(train_dir)
    print("🔍 Collecting hashes from validation set...")
    val_hashes = collect_hashes(val_dir)
    print("🔍 Collecting hashes from test set...")
    test_hashes = collect_hashes(test_dir)

    plot_distribution(train_hashes, "Train Distribution")
    plot_distribution(val_hashes, "Validation Distribution")
    plot_distribution(test_hashes, "Test Distribution")

    detect_duplicates(train_hashes, test_hashes, val_hashes)

def split_dataset(source_dir, train_dir, val_dir, test_dir, ratio=(5, 2, 3)):
    import shutil
    from pathlib import Path

    for category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, category)
        if not os.path.isdir(category_path):
            continue

        all_files = [f for f in os.listdir(category_path) if f.endswith(".wav")]
        np.random.shuffle(all_files)

        total = len(all_files)
        train_end = int(ratio[0] / sum(ratio) * total)
        val_end = train_end + int(ratio[1] / sum(ratio) * total)

        train_category_dir = os.path.join(train_dir, category)
        val_category_dir = os.path.join(val_dir, category)
        test_category_dir = os.path.join(test_dir, category)

        Path(train_category_dir).mkdir(parents=True, exist_ok=True)
        Path(val_category_dir).mkdir(parents=True, exist_ok=True)
        Path(test_category_dir).mkdir(parents=True, exist_ok=True)

        for i, f in enumerate(all_files):
            src = os.path.join(category_path, f)
            if i < train_end:
                dst = os.path.join(train_category_dir, f)
            elif i < val_end:
                dst = os.path.join(val_category_dir, f)
            else:
                dst = os.path.join(test_category_dir, f)
            shutil.copy2(src, dst)

if __name__ == "__main__":
    # 자동 분할 수행 (비율 5:2:3)
    split_dataset("sound/", "sound/train", "sound/val", "sound/test")

    # 분할된 데이터셋 확인
    run_check("sound/train", "sound/val", "sound/test")
