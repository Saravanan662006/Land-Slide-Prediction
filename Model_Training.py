import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
# 1. LOAD DATASET
# ============================================================

FILE_PATH = r"C:\Users\SARAVANAN\Desktop\SIH\Code\improved_landslide_GRU_dataset_100_events.csv"

df = pd.read_csv(FILE_PATH)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. SORT BY TIME
# ============================================================

df["datetime_ist"] = pd.to_datetime(
    df["datetime_ist"],
    format="mixed",
    dayfirst=True
)

df = df.sort_values("datetime_ist").reset_index(drop=True)


# ============================================================
# 3. REMOVE ROWS WITHOUT TARGET
# ============================================================

df = df.dropna(subset=["landslide_next_6h"]).reset_index(drop=True)


# ============================================================
# 4. CREATE ADDITIONAL SENSOR FEATURES
# ============================================================

# Acceleration magnitude
df["accel_magnitude"] = np.sqrt(
    df["accel_x_ms2"] ** 2 +
    df["accel_y_ms2"] ** 2 +
    df["accel_z_ms2"] ** 2
)

# Gyroscope magnitude
df["gyro_magnitude"] = np.sqrt(
    df["gyro_x_dps"] ** 2 +
    df["gyro_y_dps"] ** 2 +
    df["gyro_z_dps"] ** 2
)


# ============================================================
# 5. SELECT INPUT FEATURES
# ============================================================

FEATURES = [
    # Rainfall
    "cwc_rainfall_mm",
    "rainfall_6h_mm",
    "rainfall_24h_mm",
    "rainfall_72h_mm",

    # Pore-water pressure
    "pore_pressure_kpa",
    "pressure_change_6h_kpa",
    "pressure_change_24h_kpa",

    # Ground movement
    "tilt_deg",
    "pitch_deg",
    "roll_deg",

    # Ground vibration / motion
    "accel_magnitude",
    "gyro_magnitude"
]

TARGET = "landslide_next_6h"


# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

df[FEATURES] = df[FEATURES].interpolate(
    method="linear",
    limit_direction="both"
)

df[FEATURES] = df[FEATURES].fillna(
    df[FEATURES].median()
)


# ============================================================
# 7. CHECK TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")
print(df[TARGET].value_counts())

print("\nPositive landslide samples:",
      int(df[TARGET].sum()))


# ============================================================
# 8. CREATE TIME SEQUENCES
# ============================================================

# Dataset is hourly.
# 24 steps = previous 24 hours.

TIME_STEPS = 24

X_raw = df[FEATURES].values
y_raw = df[TARGET].astype(int).values

sequences = []
targets = []

for i in range(TIME_STEPS, len(df)):

    sequence = X_raw[i - TIME_STEPS:i]

    target = y_raw[i]

    sequences.append(sequence)
    targets.append(target)

X = np.array(sequences)
y = np.array(targets)

print("\nGRU input shape:", X.shape)
print("GRU target shape:", y.shape)


# ============================================================
# 9. SCALE DATA
# ============================================================

# Reshape temporarily so StandardScaler can work
samples, timesteps, features = X.shape

X_2d = X.reshape(-1, features)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_2d)

X_scaled = X_scaled.reshape(
    samples,
    timesteps,
    features
)


# ============================================================
# 10. STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# IMPORTANT:
# Your dataset has only 12 positive examples.
# A purely chronological split would leave almost no
# positive examples in the training set.
#
# Therefore this split is suitable for a DEMONSTRATION,
# not a true field-performance evaluation.

X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print("\nTraining:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("\nTraining positives:", y_train.sum())
print("Validation positives:", y_val.sum())
print("Testing positives:", y_test.sum())


# ============================================================
# 11. HANDLE CLASS IMBALANCE
# ============================================================

classes = np.unique(y_train)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weight_dict = {
    int(classes[i]): class_weights[i]
    for i in range(len(classes))
}

print("\nClass weights:")
print(class_weight_dict)


# ============================================================
# 12. BUILD GRU MODEL
# ============================================================

model = Sequential([

    GRU(
        64,
        input_shape=(TIME_STEPS, len(FEATURES)),
        return_sequences=True
    ),

    Dropout(0.30),

    GRU(
        32,
        return_sequences=False
    ),

    Dropout(0.30),

    Dense(16, activation="relu"),

    Dense(1, activation="sigmoid")
])


# ============================================================
# 13. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(
            name="auc",
            curve="PR"
        )
    ]
)


model.summary()


# ============================================================
# 14. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6
)


# ============================================================
# 15. TRAIN GRU
# ============================================================

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=100,

    batch_size=32,

    class_weight=class_weight_dict,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# 16. EVALUATE MODEL
# ============================================================

print("\nEvaluating model...")

results = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

for name, value in zip(
    model.metrics_names,
    results
):
    print(f"{name}: {value:.4f}")


# ============================================================
# 17. PREDICTIONS
# ============================================================

probabilities = model.predict(
    X_test
).ravel()

# Default threshold
THRESHOLD = 0.50

predictions = (
    probabilities >= THRESHOLD
).astype(int)


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 20. ROC-AUC AND PR-AUC
# ============================================================

if len(np.unique(y_test)) == 2:

    roc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    print("\nROC-AUC:", round(roc, 4))
    print("PR-AUC:", round(pr_auc, 4))


# ============================================================
# 21. TRAINING GRAPHS
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GRU Training vs Validation Loss")
plt.legend()

plt.show()


plt.figure(figsize=(10, 5))

plt.plot(
    history.history["auc"],
    label="Training PR-AUC"
)

plt.plot(
    history.history["val_auc"],
    label="Validation PR-AUC"
)

plt.xlabel("Epoch")
plt.ylabel("PR-AUC")
plt.title("GRU Training vs Validation PR-AUC")
plt.legend()

plt.show()


# ============================================================
# 22. SAVE MODEL
# ============================================================

model.save(
    "landslide_gru_model.keras"
)

print("\nModel saved as:")
print("landslide_gru_model.keras")


# ============================================================
# 23. SAVE SCALER
# ============================================================

import joblib

joblib.dump(
    scaler,
    "landslide_scaler.pkl"
)

print("Scaler saved as:")
print("landslide_scaler.pkl")
