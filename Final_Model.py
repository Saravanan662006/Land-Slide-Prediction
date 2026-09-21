import numpy as np
import pandas as pd
import joblib
import time

from tensorflow.keras.models import load_model


# ============================================================
# 1. LOAD MODEL AND SCALER
# ============================================================

MODEL_PATH = "landslide_gru_model.keras"
SCALER_PATH = "landslide_scaler.pkl"

print("Loading GRU model...")
model = load_model(MODEL_PATH)

print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)

print("Model and scaler loaded successfully!")


# ============================================================
# 2. MODEL SETTINGS
# ============================================================

TIME_STEPS = 24


# IMPORTANT:
# These MUST be the exact 12 features and exact same order
# that were used while training your GRU.

FEATURE_COLUMNS = [
    "cwc_rainfall_mm",
    "rainfall_6h_mm",
    "rainfall_24h_mm",
    "rainfall_72h_mm",
    "pore_pressure_kpa",
    "pressure_change_6h_kpa",
    "pressure_change_24h_kpa",
    "tilt_deg",
    "pitch_deg",
    "roll_deg",
    "accel_z_ms2",
    "gyro_x_dps"
]


# ============================================================
# 3. LOAD LIVE DATA
# ============================================================

DATA_PATH = "improved_landslide_GRU_dataset_100_events.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded.")
print("Rows:", len(df))


# ============================================================
# 4. CHECK FEATURES
# ============================================================

missing_features = [
    col for col in FEATURE_COLUMNS
    if col not in df.columns
]

if missing_features:

    print("\nERROR!")
    print("Missing features:")

    for col in missing_features:
        print("-", col)

    raise SystemExit


# ============================================================
# 5. CREATE 24-HOUR BUFFER
# ============================================================

buffer = []


# ============================================================
# 6. RISK FUNCTION
# ============================================================

def get_risk_level(probability):

    if probability < 0.30:
        return "NORMAL"

    elif probability < 0.60:
        return "WATCH"

    elif probability < 0.80:
        return "WARNING"

    else:
        return "HIGH RISK"


# ============================================================
# 7. PROCESS LIVE SENSOR READING
# ============================================================

def predict_landslide(sensor_data):

    global buffer

    # Add latest sensor reading
    buffer.append(sensor_data)

    # Keep only latest 24 readings
    if len(buffer) > TIME_STEPS:
        buffer.pop(0)

    # Need 24 hours before prediction
    if len(buffer) < TIME_STEPS:

        remaining = TIME_STEPS - len(buffer)

        print(
            f"Collecting data... "
            f"{len(buffer)}/24 readings "
            f"({remaining} remaining)"
        )

        return None

    # --------------------------------------------------------
    # Convert buffer to numpy array
    # --------------------------------------------------------

    sequence = np.array(buffer, dtype=np.float32)

    # Shape should be:
    # (24, 12)

    print("\nSequence shape:", sequence.shape)

    # --------------------------------------------------------
    # Apply SAME scaler used during training
    # --------------------------------------------------------

    sequence_scaled = scaler.transform(sequence)

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    X = np.expand_dims(sequence_scaled, axis=0)

    # Shape:
    # (1, 24, 12)

    # --------------------------------------------------------
    # GRU prediction
    # --------------------------------------------------------

    probability = float(model.predict(X, verbose=0)[0][0])

    risk = get_risk_level(probability)

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("\n======================================")
    print("      LANDSLIDE PREDICTION")
    print("======================================")

    print(f"Probability : {probability:.4f}")
    print(f"Probability : {probability * 100:.2f}%")
    print(f"Risk Level  : {risk}")

    print("======================================")

    if risk == "HIGH RISK":

        print("🚨 LANDSLIDE WARNING 🚨")

    elif risk == "WARNING":

        print("⚠️ WARNING: Increased landslide risk")

    elif risk == "WATCH":

        print("🟡 WATCH: Monitor slope conditions")

    else:

        print("🟢 NORMAL: No immediate warning")

    return probability


# ============================================================
# 8. SIMULATE LIVE SENSOR STREAM
# ============================================================

print("\nStarting live prediction...")
print("Reading one sensor record every second.\n")


for index, row in df.iterrows():

    sensor_data = [
        row[col]
        for col in FEATURE_COLUMNS
    ]

    predict_landslide(sensor_data)

    time.sleep(1)
