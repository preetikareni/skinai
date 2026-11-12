import os, shutil, cv2

SRC = "dataset"
REJ = "dataset_rejected"
CLASSES = ["dry", "normal", "oily"]   # acne/darkcircles untouched

cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

for cls in CLASSES:
    folder = os.path.join(SRC, cls)
    rej_folder = os.path.join(REJ, cls)
    os.makedirs(rej_folder, exist_ok=True)
    moved = kept = 0

    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        img = cv2.imread(path)
        ok = False
        if img is not None:
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            # exactly 1 face, and big enough (not a far-away shot)
            if len(faces) == 1:
                fx, fy, fw, fh = faces[0]
                if fw > 0.30 * w:
                    ok = True
        if ok:
            kept += 1
        else:
            shutil.move(path, os.path.join(rej_folder, f))
            moved += 1
    print(f"{cls}: kept {kept} | moved {moved}")