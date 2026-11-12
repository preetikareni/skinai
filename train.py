import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ─── Config ───────────────────────────────────────────────────────────────────
DATASET_DIR    = "dataset"
MODEL_DIR      = "model"
MODEL_PATH     = os.path.join(MODEL_DIR, "skin_model.h5")
CLASS_MAP_PATH = os.path.join(MODEL_DIR, "class_indices.json")
IMG_SIZE       = (224, 224)   # changed: MobileNetV2 works best at 224
BATCH_SIZE     = 16
EPOCHS         = 8

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(f"Dataset directory '{DATASET_DIR}' not found!")

classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
print(f"✅ Found {len(classes)} classes: {classes}")
if len(classes) == 0:
    raise ValueError("No class subfolders found in dataset/")

for cls in classes:
    cls_path = os.path.join(DATASET_DIR, cls)
    imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
    print(f"   [{cls}]: {len(imgs)} images")

# ─── Data (preprocess_input instead of rescale) ───────────────────────────────
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    shear_range=0.1,
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
)

print("\n📂 Loading training data...")
train_data = train_datagen.flow_from_directory(
    DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training', shuffle=True,
)

print("📂 Loading validation data...")
val_data = val_datagen.flow_from_directory(
    DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', shuffle=False,
)

num_classes = train_data.num_classes
print(f"\n✅ Classes detected: {train_data.class_indices}")
print(f"   Training samples  : {train_data.samples}")
print(f"   Validation samples: {val_data.samples}")

# ─── Save class indices ───────────────────────────────────────────────────────
idx_to_class = {str(v): k for k, v in train_data.class_indices.items()}
with open(CLASS_MAP_PATH, "w") as f:
    json.dump(idx_to_class, f, indent=2)
print(f"✅ Class map saved to: {CLASS_MAP_PATH}")

# ─── Class weights (helps if some classes have fewer images) ──────────────────
counts = np.bincount(train_data.classes)
total = counts.sum()
class_weights = {i: total / (num_classes * c) for i, c in enumerate(counts)}
print(f"⚖️ Class weights: {class_weights}")

# ─── Model: MobileNetV2 (pretrained) ──────────────────────────────────────────
def build_model(num_classes):
    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
    base.trainable = False   # freeze for now

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

model = build_model(num_classes)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# ─── Callbacks ────────────────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
]

# ─── Train ────────────────────────────────────────────────────────────────────
print("\n🚀 Starting training...\n")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1,
)

# ─── Fine-tune: unfreeze top layers ───
# ─── Fine-tune: unfreeze top layers ───────────────────────────────────────────
print("\n🔧 Fine-tuning top layers...")
best_val_acc = max(history.history.get('val_accuracy', [0]))

base = model.layers[1]          # the MobileNetV2 part
base.trainable = True
for layer in base.layers[:-60]: # keep the bottom frozen
    layer.trainable = False
for layer in base.layers:       # keep BatchNorm frozen (more stable)
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

ft_callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True,
                    initial_value_threshold=best_val_acc, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1),
]

history_ft = model.fit(
    train_data,
    validation_data=val_data,
    epochs=12,
    callbacks=ft_callbacks,
    class_weight=class_weights,
    verbose=1,
)

best_ft = max(history_ft.history.get('val_accuracy', [0]))
print(f"\n🏆 Best validation accuracy: {max(best_val_acc, best_ft):.2%}")
print(f"✅ Best model saved to: {MODEL_PATH}")
print("✅ Training complete!")