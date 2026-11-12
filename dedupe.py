import os, shutil

for cls in ["acne", "darkcircles", "dry", "normal", "oily"]:
    src = os.path.join("dataset", cls)
    dst = os.path.join("dataset_dupes", cls)
    os.makedirs(dst, exist_ok=True)
    seen = set()
    moved = 0
    for f in sorted(os.listdir(src)):
        if ".rf." not in f:
            continue
        key = f.split(".rf.")[0]
        if key in seen:
            shutil.move(os.path.join(src, f), os.path.join(dst, f))
            moved += 1
        else:
            seen.add(key)
    print(cls, "moved", moved, "| left", len(os.listdir(src)))