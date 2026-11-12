import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model("model/skin_model.h5")

gen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)
val_data = gen.flow_from_directory(
    "dataset", target_size=(224, 224), batch_size=32,
    class_mode="categorical", subset="validation", shuffle=False,
)

preds = model.predict(val_data)
y_pred = preds.argmax(axis=1)
y_true = val_data.classes
names = list(val_data.class_indices.keys())
n = len(names)

# ─── Confusion matrix ─────────────────────────────────────────────────────────
cm = np.zeros((n, n), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t][p] += 1

print("\nRows = TRUE class, Columns = PREDICTED class")
print(" " * 12 + "".join(f"{x[:8]:>10}" for x in names))
for i, name in enumerate(names):
    print(f"{name:<12}" + "".join(f"{v:>10}" for v in cm[i]))

# ─── Precision, recall, F1 per class ──────────────────────────────────────────
print("\nPer-class metrics:")
print(f"{'class':<14}{'precision':>10}{'recall':>10}{'f1':>10}{'images':>10}")
precisions, recalls, f1s = [], [], []
for i, name in enumerate(names):
    tp = cm[i][i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    print(f"{name:<14}{precision:>10.2f}{recall:>10.2f}{f1:>10.2f}{cm[i].sum():>10}")

accuracy = np.trace(cm) / cm.sum()
print(f"{'macro avg':<14}{np.mean(precisions):>10.2f}{np.mean(recalls):>10.2f}{np.mean(f1s):>10.2f}{cm.sum():>10}")
print(f"\nOverall accuracy: {accuracy:.2%}")

# ─── Save confusion matrix as a picture (for the README) ──────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix (accuracy {accuracy:.1%})")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=150)
    print("\nSaved: confusion_matrix.png")
except ImportError:
    print("\nmatplotlib not installed. Run: pip install matplotlib")