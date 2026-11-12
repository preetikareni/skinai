import os, cv2

cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

for cls in ["acne", "darkcircles"]:
    folder = os.path.join("dataset", cls)
    bad = 0
    files = os.listdir(folder)
    for f in files:
        img = cv2.imread(os.path.join(folder, f))
        ok = False
        if img is not None:
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
            ok = len(faces) == 1 and faces[0][2] > 0.25 * w
        if not ok:
            bad += 1
    print(f"{cls}: total {len(files)} | would reject {bad}")