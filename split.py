

import random
import shutil
from pathlib import Path
from tqdm import tqdm


SOURCE_ROOT = Path(r"D:\oakeshott_from_net")
DEST_ROOT   = Path(r"D:\Oakeshott Project")

# Which type and subtype folder to process this run.
# TYPE_NAME    — the parent folder,  e.g. "Type_XII"
# SUBTYPE_NAME — the subtype folder, e.g. "Type_XIIa"
#                set same as TYPE_NAME for types without subtypes,
#                e.g. TYPE_NAME = "Type_XIV", SUBTYPE_NAME = "Type_XIV"

TYPE_NAME    = "Type_XII"
SUBTYPE_NAME = "Type_XII"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Accepted image formats
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


RANDOM_SEED = 42

def get_images(folder: Path) -> list[Path]:

    return sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ])


def split_files(files: list[Path], train_r: float, val_r: float
                ) -> tuple[list, list, list]:

    n = len(files)
    shuffled = files.copy()
    random.shuffle(shuffled)

    n_val  = max(1, round(n * val_r))
    n_test = max(1, round(n * (1 - train_r - val_r)))
    n_train = n - n_val - n_test

    if n_train < 1:
        # Too few images to split properly — put everything in train
        return shuffled, [], []

    train = shuffled[:n_train]
    val   = shuffled[n_train:n_train + n_val]
    test  = shuffled[n_train + n_val:]

    return train, val, test


def copy_files(files: list[Path], dest_dir: Path):
    if not files:
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dest_dir / f.name)



def main():
    random.seed(RANDOM_SEED)

    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
        "Train/val/test ratios must sum to 1.0"

    subtype_dir = SOURCE_ROOT / TYPE_NAME / SUBTYPE_NAME

    if not subtype_dir.exists():
        print(f"[ERROR] Folder not found: {subtype_dir}")
        return

    images = get_images(subtype_dir)

    print(f"Folder : {subtype_dir}")
    print(f"Dest   : {DEST_ROOT}")
    print(f"Images : {len(images)} found")
    print(f"Split  : {TRAIN_RATIO:.0%} train / {VAL_RATIO:.0%} val / {TEST_RATIO:.0%} test")
    print(f"Seed   : {RANDOM_SEED}")

    if len(images) == 0:
        print("[ERROR] No images found in folder.")
        return

    train_files, val_files, test_files = split_files(images, TRAIN_RATIO, VAL_RATIO)

    copy_files(train_files, DEST_ROOT / "train" / TYPE_NAME / SUBTYPE_NAME)
    copy_files(val_files,   DEST_ROOT / "val"   / TYPE_NAME / SUBTYPE_NAME)
    copy_files(test_files,  DEST_ROOT / "test"  / TYPE_NAME / SUBTYPE_NAME)

    print(f"\n  Train : {len(train_files)}")
    print(f"  Val   : {len(val_files)}")
    print(f"  Test  : {len(test_files)}")
    print(f"  Total : {len(images)}")
    print(f"\nDone. Add renders manually to:")
    print(f"  {DEST_ROOT / 'train' / TYPE_NAME / SUBTYPE_NAME}")


if __name__ == "__main__":
    main()