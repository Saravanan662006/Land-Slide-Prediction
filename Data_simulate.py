import numpy as np
import pandas as pd

np.random.seed(42)

start = "2024-01-01 00:00:00"
periods = 2 * 365 * 24
ts = pd.date_range(start=start, periods=periods, freq="h", tz="Asia/Kolkata")
n = len(ts)

doy = ts.dayofyear.to_numpy()
monsoon = np.exp(-0.5 * ((doy - 225) / 55) ** 2)

rain_prob = 0.025 + 0.13 * monsoon
rain_event = np.random.random(n) < rain_prob
rainfall = np.zeros(n)
rainfall[rain_event] = np.random.gamma(1.7, 5.0, rain_event.sum())
rainfall = np.clip(rainfall, 0, 80)

rain_6h = pd.Series(rainfall).rolling(6, min_periods=1).sum().to_numpy()
rain_24h = pd.Series(rainfall).rolling(24, min_periods=1).sum().to_numpy()
rain_72h = pd.Series(rainfall).rolling(72, min_periods=1).sum().to_numpy()

pressure = np.zeros(n)
pressure[0] = 1.2
for i in range(1, n):
    infiltration = 0.018 * rainfall[i] + 0.004 * rain_6h[i] + 0.0012 * rain_24h[i]
    drainage = 0.018 * pressure[i-1]
    pressure[i] = pressure[i-1] + infiltration - drainage + np.random.normal(0, 0.025)
pressure = np.clip(pressure, 0.15, 9.6)

pressure_rate = np.gradient(pressure)
pressure_24h_change = pressure - np.roll(pressure, 24)
pressure_24h_change[:24] = pressure[:24] - pressure[0]

tilt_drive = 0.20 * np.maximum(pressure - 3.0, 0) + 0.018 * np.maximum(rain_24h - 60, 0)
tilt_deg = 2.0 + np.cumsum(0.002 * tilt_drive + np.random.normal(0, 0.006, n))
tilt_deg = np.clip(tilt_deg, -2, 18)

pitch_deg = 1.2 + 0.45*np.sin(np.arange(n)/400) + 0.75*np.maximum(tilt_drive, 0)
roll_deg = 0.8 + 0.35*np.sin(np.arange(n)/250) + 0.50*np.maximum(tilt_drive, 0)
yaw_deg = 5 + 0.8*np.sin(np.arange(n)/700) + np.random.normal(0, 0.12, n)

accel_x = np.random.normal(0, 0.015, n) + 0.010*np.gradient(tilt_deg)
accel_y = np.random.normal(0, 0.015, n) + 0.008*np.gradient(pitch_deg)
accel_z = 9.81 + np.random.normal(0, 0.025, n)

gyro_x = np.random.normal(0, 0.08, n) + 0.35*np.gradient(roll_deg)
gyro_y = np.random.normal(0, 0.08, n) + 0.35*np.gradient(pitch_deg)
gyro_z = np.random.normal(0, 0.08, n) + 0.20*np.gradient(yaw_deg)

pressure_score = np.clip((pressure - 4.5)/4.0, 0, 1)
rain_score = np.clip((rain_72h - 100)/180, 0, 1)
tilt_score = np.clip((np.abs(np.gradient(tilt_deg)) - 0.015)/0.08, 0, 1)

risk_score = np.clip(
    0.45*pressure_score + 0.35*rain_score + 0.20*tilt_score
    + np.random.normal(0, 0.025, n), 0, 1
)

landslide_event = (
    (pressure > 7.2) &
    (rain_72h > 130) &
    ((pressure_rate > 0.08) | (np.abs(np.gradient(tilt_deg)) > 0.025))
).astype(int)

risk_level = np.select(
    [risk_score < 0.25, risk_score < 0.50, risk_score < 0.75],
    ["LOW", "MODERATE", "HIGH"],
    default="CRITICAL"
)

df = pd.DataFrame({
    "datetime_ist": ts.strftime("%Y-%m-%d %H:%M:%S"),
    "date": ts.strftime("%Y-%m-%d"),
    "time": ts.strftime("%H:%M:%S"),
    "cwc_rainfall_mm": np.round(rainfall, 2),
    "rainfall_6h_mm": np.round(rain_6h, 2),
    "rainfall_24h_mm": np.round(rain_24h, 2),
    "rainfall_72h_mm": np.round(rain_72h, 2),
    "pore_pressure_kpa": np.round(pressure, 3),
    "pore_pressure_mbar": np.round(pressure*10, 2),
    "pressure_change_24h_kpa": np.round(pressure_24h_change, 3),
    "tilt_deg": np.round(tilt_deg, 3),
    "pitch_deg": np.round(pitch_deg, 3),
    "roll_deg": np.round(roll_deg, 3),
    "yaw_deg": np.round(yaw_deg, 3),
    "accel_x_ms2": np.round(accel_x, 4),
    "accel_y_ms2": np.round(accel_y, 4),
    "accel_z_ms2": np.round(accel_z, 4),
    "gyro_x_dps": np.round(gyro_x, 4),
    "gyro_y_dps": np.round(gyro_y, 4),
    "gyro_z_dps": np.round(gyro_z, 4),
    "landslide_risk_score": np.round(risk_score, 4),
    "landslide_risk_level": risk_level,
    "landslide_event": landslide_event
})

df.to_csv("synthetic_landslide_GRU_dataset_2years.csv", index=False)
print(df.head())
print("Rows:", len(df))
print("Landslide events:", df["landslide_event"].sum())
