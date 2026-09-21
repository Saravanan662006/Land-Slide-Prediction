import numpy as np
import pandas as pd

# ============================================================
# SYNTHETIC LANDSLIDE DATASET GENERATOR
# Honeywell SSC + BNO085 + CWC rainfall -> GRU prediction
# ============================================================

np.random.seed(42)

# ------------------------------------------------------------
# 1. TIME
# ------------------------------------------------------------

START_DATE = "2024-01-01 00:00:00"
YEARS = 2
HOURS = YEARS * 365 * 24

time_index = pd.date_range(
    start=START_DATE,
    periods=HOURS,
    freq="h",
    tz="Asia/Kolkata"
)

n = len(time_index)

doy = time_index.dayofyear.to_numpy()


# ============================================================
# 2. SYNTHETIC CWC-LIKE RAINFALL
# ============================================================
# NOTE:
# This rainfall is SIMULATED.
# Later replace it with actual CWC rainfall data.

# Monsoon season effect
monsoon_strength = np.exp(
    -0.5 * ((doy - 225) / 55) ** 2
)

rain_probability = (
    0.015
    + 0.14 * monsoon_strength
)

rain_occurs = (
    np.random.random(n) < rain_probability
)

rainfall = np.zeros(n)

# Random rainfall intensity
rainfall[rain_occurs] = np.random.gamma(
    shape=1.8,
    scale=4.5,
    size=rain_occurs.sum()
)

# ------------------------------------------------------------
# Add prolonged rainfall/storm events
# ------------------------------------------------------------

storm_centers = np.random.choice(
    np.arange(720, n - 720),
    size=30,
    replace=False
)

for center in storm_centers:

    # Mostly generate storms during monsoon
    if 120 <= doy[center] <= 285:

        duration = np.random.randint(8, 48)

        idx = np.arange(
            center,
            min(center + duration, n)
        )

        intensity = np.random.uniform(3, 11)

        phase = (
            (idx - center)
            / max(duration - 1, 1)
        )

        storm_profile = (
            np.sin(np.pi * phase) ** 1.4
        )

        rainfall[idx] += (
            intensity * storm_profile
        )


# Small measurement variability
rainfall += np.random.normal(
    0,
    0.12,
    n
)

rainfall = np.clip(
    rainfall,
    0,
    100
)


# ============================================================
# 3. RAINFALL ACCUMULATION
# ============================================================

rain_series = pd.Series(rainfall)

rainfall_6h = (
    rain_series
    .rolling(6, min_periods=1)
    .sum()
    .to_numpy()
)

rainfall_24h = (
    rain_series
    .rolling(24, min_periods=1)
    .sum()
    .to_numpy()
)

rainfall_72h = (
    rain_series
    .rolling(72, min_periods=1)
    .sum()
    .to_numpy()
)


# ============================================================
# 4. HONEYWELL PORE PRESSURE
# ============================================================
#
# Rainfall
#     ↓
# Infiltration
#     ↓
# Pore pressure
#
# Pressure does NOT instantly equal rainfall.
# It has:
#   - infiltration
#   - accumulation
#   - drainage
#   - noise
#

pressure = np.zeros(n)

# Initial pressure
pressure[0] = 0.8

for i in range(1, n):

    # Direct rainfall infiltration
    direct_infiltration = (
        0.010 * rainfall[i]
    )

    # Short-term accumulated rainfall
    short_term = (
        0.0035 * rainfall_6h[i]
    )

    # Longer-term accumulated rainfall
    long_term = (
        0.0010 * rainfall_24h[i]
    )

    # Natural drainage
    drainage = (
        0.025 * pressure[i - 1]
    )

    # Sensor / soil noise
    noise = np.random.normal(
        0,
        0.018
    )

    pressure[i] = (
        pressure[i - 1]
        + direct_infiltration
        + short_term
        + long_term
        - drainage
        + noise
    )


# Honeywell SSC 100MG:
# 0-100 mbar = 0-10 kPa
pressure = np.clip(
    pressure,
    0.15,
    9.7
)


# ============================================================
# 5. PRESSURE CHANGE
# ============================================================

pressure_series = pd.Series(pressure)

pressure_change_1h = np.gradient(
    pressure
)

pressure_change_6h = (
    pressure
    - pressure_series.shift(6)
    .bfill()
    .to_numpy()
)

pressure_change_24h = (
    pressure
    - pressure_series.shift(24)
    .bfill()
    .to_numpy()
)


# ============================================================
# 6. HIDDEN SLOPE SUSCEPTIBILITY
# ============================================================
#
# This prevents:
#
#     HIGH PRESSURE = LANDSLIDE ALWAYS
#
# Different slopes can react differently.

susceptibility = np.zeros(n)

susceptibility[0] = 0.45

for i in range(1, n):

    susceptibility[i] = (
        0.995 * susceptibility[i - 1]
        + 0.005 * np.random.uniform(
            0.25,
            0.75
        )
    )


# ============================================================
# 7. DETERMINE WHEN GROUND MOVEMENT CAN OCCUR
# ============================================================

# High pressure indicator
pressure_high = np.clip(
    (pressure - 5.0) / 4.0,
    0,
    1
)

# Pressure increasing indicator
pressure_rising = np.clip(
    (pressure_change_6h - 0.03) / 0.25,
    0,
    1
)

# Heavy accumulated rainfall
rain_stress = np.clip(
    (rainfall_72h - 80) / 180,
    0,
    1
)


# ------------------------------------------------------------
# Movement probability
# ------------------------------------------------------------
#
# Important:
# Even when pressure is HIGH,
# movement is NOT guaranteed.

movement_probability = (
    0.015
    + 0.45 * pressure_high
    + 0.30 * pressure_rising
    + 0.20 * rain_stress
    + 0.20 * susceptibility
)

movement_probability = np.clip(
    movement_probability,
    0.005,
    0.80
)


# Randomly decide whether movement occurs
movement_active = (
    np.random.random(n)
    < movement_probability
)


# Strong movement is much less frequent
strong_movement = (
    movement_active
    &
    (
        np.random.random(n)
        <
        (0.10 + 0.35 * pressure_high)
    )
)


# ============================================================
# 8. BNO085 MOVEMENT MAGNITUDE
# ============================================================

movement = np.zeros(n)

# Small movements
movement[movement_active] = (
    np.random.uniform(
        0.02,
        0.10,
        movement_active.sum()
    )
)

# Strong movements
movement[strong_movement] += (
    np.random.uniform(
        0.08,
        0.45,
        strong_movement.sum()
    )
)

# Movement direction
movement *= np.random.choice(
    [-1, 1],
    size=n
)


# ============================================================
# 9. BNO085 TILT
# ============================================================

tilt = np.zeros(n)

# Initial terrain inclination
tilt[0] = 2.0

for i in range(1, n):

    # Normal sensor/terrain drift
    natural_drift = np.random.normal(
        0,
        0.006
    )

    # Movement only affects tilt when movement is active
    tilt[i] = (
        tilt[i - 1]
        + natural_drift
        + movement[i]
    )

    tilt[i] = np.clip(
        tilt[i],
        -5,
        25
    )


# ============================================================
# 10. BNO085 PITCH / ROLL / YAW
# ============================================================

pitch = (
    1.2
    + 0.12 * np.sin(
        np.arange(n) / 300
    )
    + 0.55 * movement
    + np.random.normal(
        0,
        0.025,
        n
    )
)

roll = (
    0.8
    + 0.10 * np.sin(
        np.arange(n) / 250
    )
    + 0.40 * movement
    + np.random.normal(
        0,
        0.025,
        n
    )
)

yaw = (
    5.0
    + 0.6 * np.sin(
        np.arange(n) / 700
    )
    + 0.15 * movement
    + np.random.normal(
        0,
        0.08,
        n
    )
)


# ============================================================
# 11. BNO085 ACCELEROMETER
# ============================================================

accel_x = (
    np.random.normal(
        0,
        0.012,
        n
    )
    + 0.20 * movement
)

accel_y = (
    np.random.normal(
        0,
        0.012,
        n
    )
    + 0.16 * movement
)

accel_z = (
    9.81
    + np.random.normal(
        0,
        0.025,
        n
    )
    + 0.25 * np.abs(movement)
)


# ============================================================
# 12. BNO085 GYROSCOPE
# ============================================================

gyro_x = (
    np.random.normal(
        0,
        0.04,
        n
    )
    + 0.30 * np.gradient(roll)
)

gyro_y = (
    np.random.normal(
        0,
        0.04,
        n
    )
    + 0.30 * np.gradient(pitch)
)

gyro_z = (
    np.random.normal(
        0,
        0.04,
        n
    )
    + 0.15 * np.gradient(yaw)
)


# Occasional rotational disturbances
gyro_x[strong_movement] += np.random.normal(
    0,
    0.20,
    strong_movement.sum()
)

gyro_y[strong_movement] += np.random.normal(
    0,
    0.20,
    strong_movement.sum()
)


# ============================================================
# 13. LANDSLIDE RISK SCORE
# ============================================================

pressure_score = np.clip(
    (pressure - 5.5) / 3.5,
    0,
    1
)

pressure_rise_score = np.clip(
    (pressure_change_6h - 0.05) / 0.30,
    0,
    1
)

rainfall_score = np.clip(
    (rainfall_72h - 100) / 220,
    0,
    1
)

movement_score = np.clip(
    np.abs(movement) / 0.45,
    0,
    1
)


# Combined risk
risk_score = (
    0.40 * pressure_score
    + 0.20 * pressure_rise_score
    + 0.20 * rainfall_score
    + 0.15 * movement_score
    + 0.05 * susceptibility
)

# Small uncertainty
risk_score += np.random.normal(
    0,
    0.018,
    n
)

risk_score = np.clip(
    risk_score,
    0,
    1
)


# ============================================================
# 14. SIMULATED LANDSLIDE EVENTS
# ============================================================
#
# An event requires multiple conditions.
#
# It does NOT happen simply because pressure crosses a threshold.

failure_drive = (
    0.45 * pressure_score
    + 0.25 * pressure_rise_score
    + 0.15 * rainfall_score
    + 0.15 * movement_score
)

failure_probability = np.clip(
    (failure_drive - 0.68) * 0.65,
    0,
    0.18
)

landslide_event = (
    np.random.random(n)
    < failure_probability
)

# Require meaningful rainfall and pressure.
landslide_event = (
    landslide_event
    & (pressure > 6.0)
    & (rainfall_72h > 80)
)

landslide_event = (
    landslide_event
    .astype(int)
)


# ============================================================
# 15. RISK LEVEL
# ============================================================

risk_level = np.select(
    [
        risk_score < 0.25,
        risk_score < 0.50,
        risk_score < 0.75
    ],
    [
        "LOW",
        "MODERATE",
        "HIGH"
    ],
    default="CRITICAL"
)


# ============================================================
# 16. GRU TARGET
# ============================================================
#
# Predict:
#
# "Will a landslide happen during
#  the following 6 hours?"
#

landslide_next_6h = np.full(
    n,
    np.nan
)

for i in range(n - 6):

    future_events = landslide_event[
        i + 1 : i + 7
    ]

    landslide_next_6h[i] = int(
        future_events.max()
    )


# ============================================================
# 17. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame({

    # Time
    "datetime_ist":
        time_index.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

    "date":
        time_index.strftime(
            "%Y-%m-%d"
        ),

    "time":
        time_index.strftime(
            "%H:%M:%S"
        ),

    # Rainfall
    "cwc_rainfall_mm":
        np.round(
            rainfall,
            2
        ),

    "rainfall_6h_mm":
        np.round(
            rainfall_6h,
            2
        ),

    "rainfall_24h_mm":
        np.round(
            rainfall_24h,
            2
        ),

    "rainfall_72h_mm":
        np.round(
            rainfall_72h,
            2
        ),

    # Honeywell pressure
    "pore_pressure_kpa":
        np.round(
            pressure,
            3
        ),

    "pore_pressure_mbar":
        np.round(
            pressure * 10,
            2
        ),

    "pressure_change_6h_kpa":
        np.round(
            pressure_change_6h,
            4
        ),

    "pressure_change_24h_kpa":
        np.round(
            pressure_change_24h,
            4
        ),

    # BNO085
    "tilt_deg":
        np.round(
            tilt,
            3
        ),

    "pitch_deg":
        np.round(
            pitch,
            3
        ),

    "roll_deg":
        np.round(
            roll,
            3
        ),

    "yaw_deg":
        np.round(
            yaw,
            3
        ),

    # Accelerometer
    "accel_x_ms2":
        np.round(
            accel_x,
            4
        ),

    "accel_y_ms2":
        np.round(
            accel_y,
            4
        ),

    "accel_z_ms2":
        np.round(
            accel_z,
            4
        ),

    # Gyroscope
    "gyro_x_dps":
        np.round(
            gyro_x,
            4
        ),

    "gyro_y_dps":
        np.round(
            gyro_y,
            4
        ),

    "gyro_z_dps":
        np.round(
            gyro_z,
            4
        ),

    # ML outputs
    "landslide_risk_score":
        np.round(
            risk_score,
            4
        ),

    "landslide_risk_level":
        risk_level,

    "landslide_event":
        landslide_event,

    # GRU target
    "landslide_next_6h":
        landslide_next_6h
})


# ============================================================
# 18. SAVE DATASET
# ============================================================

OUTPUT_FILE = (
    "synthetic_landslide_GRU_dataset_realistic.csv"
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 19. PRINT SUMMARY
# ============================================================

print("=" * 60)
print("LANDSLIDE DATASET GENERATED")
print("=" * 60)

print(
    "Number of records:",
    len(df)
)

print(
    "Number of columns:",
    len(df.columns)
)

print(
    "Simulated landslide events:",
    int(landslide_event.sum())
)

print(
    "Movement-active records:",
    int(movement_active.sum())
)

print(
    "Strong movement records:",
    int(strong_movement.sum())
)

print(
    "Maximum pore pressure:",
    round(
        pressure.max(),
        2
    ),
    "kPa"
)

print(
    "Average pore pressure:",
    round(
        pressure.mean(),
        2
    ),
    "kPa"
)

print("\nFirst 10 rows:")
print(
    df.head(10).to_string(
        index=False
    )
)

print(
    "\nDataset saved as:",
    OUTPUT_FILE
)