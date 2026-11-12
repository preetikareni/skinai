import os, shutil
import numpy as np
from PIL import Image

NEW = "dataset_new"
NEW_DUPES = "dataset_new_dupes"
# Images already in your project. New photos that match any of these are moved out.
KNOWN = ["dataset", "dataset_dupes", "dataset_rejected"]
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAX_DIFF = 5          # 0 = identical, higher = more different (64 bits in total)


def dhash(path):
    """64-bit fingerprint of how brightness changes across the picture."""
    try:
        img = Image.open(path).convert("L").resize((9, 8))
    except Exception:
        return None
    px = np.asarray(img, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def diff(a, b):
    return bin(a ^ b).count("1")


def all_images(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(EXTS):
                yield os.path.join(root, f)


print("Reading your existing images (this takes a minute)...")
known_hashes = []
for folder in KNOWN:
    if os.path.isdir(folder):
        for p in all_images(folder):
            h = dhash(p)
            if h is not None:
                known_hashes.append(h)
print(f"Existing images fingerprinted: {len(known_hashes)}\n")

kept_hashes = []   # new images already accepted (to catch copies inside dataset_new)
for cls in sorted(os.listdir(NEW)):
    src = os.path.join(NEW, cls)
    if not os.path.isdir(src):
        continue
    dst = os.path.join(NEW_DUPES, cls)
    os.makedirs(dst, exist_ok=True)
    moved_old = moved_inside = unreadable = 0
    for f in sorted(os.listdir(src)):
        if not f.lower().endswith(EXTS):
            continue
        path = os.path.join(src, f)
        h = dhash(path)
        if h is None:
            unreadable += 1
            continue
        if any(diff(h, k) <= MAX_DIFF for k in known_hashes):
            shutil.move(path, os.path.join(dst, "old_" + f))
            moved_old += 1
        elif any(diff(h, k) <= MAX_DIFF for k in kept_hashes):
            shutil.move(path, os.path.join(dst, "same_" + f))
            moved_inside += 1
        else:
            kept_hashes.append(h)
    left = len([x for x in os.listdir(src) if x.lower().endswith(EXTS)])
    print(f"{cls}: already in your project {moved_old} | copies inside new folder {moved_inside} "
          f"| unreadable {unreadable} | left {left}")

print("\nNothing was deleted. Moved files are in dataset_new_dupes.")