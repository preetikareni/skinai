import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

FOLDER = "dataset/darkcircles"
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

model = tf.keras.models.load_model("model/skin_model.h5")

# class order must match training (folders sorted alphabetically)
classes = sorted(d for d in os.listdir("dataset")
                 if os.path.isdir(os.path.join("dataset", d)))
dc = classes.index("darkcircles")

rows = []
for fn in os.listdir(FOLDER):
    if not fn.lower().endswith(EXTS):
        continue
    path = os.path.join(FOLDER, fn)
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        print("Unreadable:", fn)
        continue
    w, h = img.size
    small = img.resize((224, 224))
    gray = np.array(small.convert("L"), dtype=np.float32)
    bright = gray.mean()
    p = model.predict(preprocess_input(np.array(small, dtype=np.float32)[None]), verbose=0)[0]
    rows.append((fn, path, classes[int(p.argmax())], float(p[dc]), bright, min(w, h)))

# Lowest darkcircles score first = most suspicious
rows.sort(key=lambda r: r[3])
print(f"Checked {len(rows)} darkcircles images\n")
print("Lowest darkcircles scores (maybe wrong or not useful):")
for fn, _, pred, score, bright, side in rows[:15]:
    print(f"  {fn[:40]:<42} looks {pred:<12} darkcircles {score:.0%}")

dark = [r for r in rows if r[4] < 50 or r[4] > 215]
small = [r for r in rows if r[5] < 100]
print(f"\nToo dark/bright: {len(dark)} | Too small: {len(small)}")

sheet = Image.new("RGB", (800, 800))
draw = ImageDraw.Draw(sheet)
for i, (fn, path, pred, score, _, _) in enumerate(rows[:16]):
    im = Image.open(path).convert("RGB").resize((200, 200))
    x, y = (i % 4) * 200, (i // 4) * 200
    sheet.paste(im, (x, y))
    draw.rectangle([x, y, x + 200, y + 14], fill=(0, 0, 0))
    draw.text((x + 3, y + 1), f"looks {pred} {score:.0%}", fill=(255, 255, 0))
sheet.save("suspect_darkcircles.jpg")
print("\nSaved: suspect_darkcircles.jpg (the 16 most suspicious images)")